//! A safetensors reader, by hand: `u64` little-endian header length, a JSON header, then the data.
//!
//! tessl has no safetensors or mmap helper (`rg -i "safetensor|mmap" tessl/src` is empty; its
//! loader is `npy.rs`), and the format is small enough that a crate for it would be a new
//! dependency for what `std` and `serde_json` already do. Reads are positioned
//! (`FileExt::read_exact_at`) straight into the destination, so a tensor bound for a GPU buffer is
//! read into the buffer's own mapping with no intermediate copy.
//!
//! Everything the header claims is checked before any tensor is read: a tensor's byte length is
//! its shape times its element size, every range lies inside the file, and no two ranges overlap.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{MetalError, Result};

/// The format's own limit on the header (safetensors spec: 100 MB).
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(Dtype::Bf16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
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
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

pub struct SafeTensors {
    path: PathBuf,
    file: File,
    tensors: BTreeMap<String, TensorInfo>,
}

fn bad(path: &Path, msg: impl std::fmt::Display) -> MetalError {
    MetalError::Weights(format!("{}: {msg}", path.display()))
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| bad(path, e))?;
        let file_len = file.metadata().map_err(|e| bad(path, e))?.len();
        if file_len < 8 {
            return Err(bad(path, format!("{file_len} bytes is too short for a header length")));
        }
        let mut len_bytes = [0u8; 8];
        file.read_exact_at(&mut len_bytes, 0).map_err(|e| bad(path, e))?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_BYTES {
            return Err(bad(path, format!("header length {header_len} exceeds {MAX_HEADER_BYTES}")));
        }
        let data_start = 8 + header_len;
        if data_start > file_len {
            return Err(bad(path, format!("header length {header_len} runs past the end of a {file_len}-byte file")));
        }
        // header_len <= 100 MB, so it fits usize.
        let mut header = vec![0u8; usize::try_from(header_len).map_err(|e| bad(path, e))?];
        file.read_exact_at(&mut header, 8).map_err(|e| bad(path, e))?;
        let tensors = parse_header(&header, data_start, file_len).map_err(|m| bad(path, m))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| bad(&self.path, format!("no tensor named {name}")))
    }

    /// The tensor's info, after checking its shape and that its dtype is one of `dtypes`.
    pub fn expect(&self, name: &str, shape: &[usize], dtypes: &[Dtype]) -> Result<&TensorInfo> {
        let info = self.info(name)?;
        if info.shape != shape {
            return Err(bad(
                &self.path,
                format!("{name} has shape {:?}, expected {shape:?}", info.shape),
            ));
        }
        if !dtypes.contains(&info.dtype) {
            return Err(bad(
                &self.path,
                format!("{name} is {:?}, expected one of {dtypes:?}", info.dtype),
            ));
        }
        Ok(info)
    }

    /// Read a tensor's raw bytes into `dst`, which must be exactly its size.
    pub fn read_into(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let info = self.info(name)?;
        if dst.len() != info.nbytes() {
            return Err(bad(
                &self.path,
                format!("{name}: {} bytes, destination holds {}", info.nbytes(), dst.len()),
            ));
        }
        self.file
            .read_exact_at(dst, info.start)
            .map_err(|e| bad(&self.path, format!("{name}: {e}")))
    }

    /// A bf16 tensor of `shape`, as raw bit patterns.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let info = self.expect(name, shape, &[Dtype::Bf16])?.clone();
        let mut bytes = vec![0u8; info.nbytes()];
        self.read_into(name, &mut bytes)?;
        Ok(u16_words(&bytes))
    }

    /// A tensor of `shape` in any of the float dtypes, widened to f32 exactly.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self
            .expect(name, shape, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?
            .clone();
        let mut bytes = vec![0u8; info.nbytes()];
        self.read_into(name, &mut bytes)?;
        Ok(widen(info.dtype, &bytes))
    }
}

/// Little-endian 16-bit words.
pub fn u16_words(bytes: &[u8]) -> Vec<u16> {
    bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect()
}

/// Little-endian `dtype` elements, widened to f32 exactly.
pub fn widen(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        Dtype::Bf16 => u16_words(bytes).into_iter().map(tessl::tensor::bf16_bits_to_f32).collect(),
        Dtype::F16 => u16_words(bytes).into_iter().map(tessl::tensor::f16_bits_to_f32).collect(),
    }
}

fn parse_header(header: &[u8], data_start: u64, file_len: u64) -> std::result::Result<BTreeMap<String, TensorInfo>, String> {
    let root: Value =
        serde_json::from_slice(header).map_err(|e| format!("header is not JSON: {e}"))?;
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
        let dtype = Dtype::parse(dtype_s)
            .ok_or_else(|| format!("{name}: dtype {dtype_s} is not one this reader loads (BF16, F16, F32)"))?;
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
            return Err(format!(
                "{name}: data_offsets [{b}, {e}) do not fit the {data_len}-byte data section"
            ));
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
    let mut ranges: Vec<(&str, u64, u64)> = tensors
        .iter()
        .map(|(n, t)| (n.as_str(), t.start, t.end))
        .collect();
    ranges.sort_by_key(|r| (r.1, r.2));
    for w in ranges.windows(2) {
        if w[1].1 < w[0].2 {
            return Err(format!("tensors {} and {} overlap", w[0].0, w[1].0));
        }
    }
    Ok(tensors)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A file in the temp dir, removed on drop.
    pub(crate) struct TempFile(pub PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            // A leftover temp file is harmless and the test already has its answer.
            let _ignored = std::fs::remove_file(&self.0);
        }
    }

    pub(crate) fn write_raw(header: &str, data: &[u8]) -> TempFile {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("qd-metal-st-{}-{n}.safetensors", std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        TempFile(path)
    }

    fn two_tensor_file() -> TempFile {
        // a: bf16 [2, 2] = 1, 2, -3, 0.5; b: f32 [3] = 1.5, -2, 7
        let mut data = Vec::new();
        for v in [1.0f32, 2.0, -3.0, 0.5] {
            data.extend_from_slice(&tessl::tensor::f32_to_bf16_bits(v).to_le_bytes());
        }
        for v in [1.5f32, -2.0, 7.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        write_raw(
            r#"{"__metadata__":{"format":"pt"},
               "a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},
               "b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}}"#,
            &data,
        )
    }

    #[test]
    fn reads_tensors_and_widens_to_f32() {
        let f = two_tensor_file();
        let st = SafeTensors::open(&f.0).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(st.f32("a", &[2, 2]).unwrap(), [1.0, 2.0, -3.0, 0.5]);
        assert_eq!(st.f32("b", &[3]).unwrap(), [1.5, -2.0, 7.0]);
        assert_eq!(st.bf16("a", &[2, 2]).unwrap()[1], tessl::tensor::f32_to_bf16_bits(2.0));
        assert!(st.bf16("b", &[3]).unwrap_err().to_string().contains("expected one of [Bf16]"));
        assert!(st.f32("a", &[4]).unwrap_err().to_string().contains("shape [2, 2]"));
        assert!(st.f32("c", &[1]).unwrap_err().to_string().contains("no tensor named c"));
        let mut small = [0u8; 7];
        assert!(st.read_into("a", &mut small).unwrap_err().to_string().contains("destination"));
    }

    #[test]
    fn refuses_malformed_headers() {
        let cases: [(&str, usize, &str); 7] = [
            ("not json", 0, "not JSON"),
            (r#"[1]"#, 0, "not a JSON object"),
            (r#"{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#, 1, "dtype I8"),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, 8, "span 4"),
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, 4, "do not fit"),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                    "b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
                8,
                "overlap",
            ),
            (r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#, 4, "not 2"),
        ];
        for (header, data_len, needle) in cases {
            let f = write_raw(header, &vec![0u8; data_len]);
            let err = SafeTensors::open(&f.0).err().expect(needle).to_string();
            assert!(err.contains(needle), "{needle:?} not in {err:?}");
        }
    }

    #[test]
    fn refuses_truncated_files() {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("qd-metal-st-short-{}-{n}", std::process::id()));
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        let f = TempFile(path);
        assert!(SafeTensors::open(&f.0).err().unwrap().to_string().contains("too short"));

        let path = std::env::temp_dir().join(format!("qd-metal-st-long-{}-{n}", std::process::id()));
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, bytes).unwrap();
        let f = TempFile(path);
        assert!(SafeTensors::open(&f.0).err().unwrap().to_string().contains("runs past"));
    }
}
