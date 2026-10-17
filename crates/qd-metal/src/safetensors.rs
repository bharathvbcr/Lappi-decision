//! The serving loader's safetensors reader: a thin adapter over `qd_export::safetensors`, which
//! owns the format (header length, the 64 MiB header bound, JSON parsing, shape/offset/overlap
//! checks; zero-length headers are refused there). What this module adds is serving-side only:
//! the three-dtype admission (this loader reads BF16, F16 and F32 and nothing else, checked when
//! the file is opened), `Refusal` -> [`MetalError`], and the typed accessors tessl's loaders use.
//!
//! tessl has had a strict safetensors reader of its own since 2026-09-29 (`tessl/src/safetensors.rs`,
//! commit 61e00c3, positioned reads, header bound 100 MB). This loader does not use it: the
//! workspace's one Rust owner of the format is `qd_export::safetensors` (ft-e0211ba7412), whose
//! 64 MiB header bound is the number the Python reader and the shared adversarial case list
//! hold, and two Rust readers with different bounds is what that card removed. Reads are
//! positioned (`FileExt::read_exact_at`, inside qd-export's reader) straight into the
//! destination, so a tensor bound for a GPU buffer is read into the buffer's own mapping with no
//! intermediate copy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qd_export::safetensors::SafeTensorsFile;

use crate::error::{MetalError, Result};

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
    file: SafeTensorsFile,
    tensors: BTreeMap<String, TensorInfo>,
}

fn bad(path: &Path, msg: impl std::fmt::Display) -> MetalError {
    MetalError::Weights(format!("{}: {msg}", path.display()))
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        // A refusal's detail already names the path.
        let file = SafeTensorsFile::open(path).map_err(|r| MetalError::Weights(r.detail))?;
        let mut tensors = BTreeMap::new();
        for (name, t) in file.tensors() {
            let dtype = Dtype::parse(t.dtype.as_str()).ok_or_else(|| {
                bad(
                    path,
                    format!(
                        "{name}: dtype {} is not one this reader loads (BF16, F16, F32)",
                        t.dtype.as_str()
                    ),
                )
            })?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape: t.shape.clone(),
                    start: t.start,
                    end: t.end,
                },
            );
        }
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
                format!(
                    "{name}: {} bytes, destination holds {}",
                    info.nbytes(),
                    dst.len()
                ),
            ));
        }
        let owned = self
            .file
            .tensors()
            .get(name)
            .ok_or_else(|| bad(&self.path, format!("no tensor named {name}")))?;
        self.file
            .read_tensor_range(owned, 0, dst)
            .map_err(|r| bad(&self.path, format!("{name}: {}", r.detail)))
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
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

/// Little-endian `dtype` elements, widened to f32 exactly.
pub fn widen(dtype: Dtype, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::Bf16 => u16_words(bytes)
            .into_iter()
            .map(tessl::tensor::bf16_bits_to_f32)
            .collect(),
        Dtype::F16 => u16_words(bytes)
            .into_iter()
            .map(tessl::tensor::f16_bits_to_f32)
            .collect(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fs::File;
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
        let path = std::env::temp_dir().join(format!(
            "qd-metal-st-{}-{n}.safetensors",
            std::process::id()
        ));
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
        assert_eq!(
            st.bf16("a", &[2, 2]).unwrap()[1],
            tessl::tensor::f32_to_bf16_bits(2.0)
        );
        assert!(
            st.bf16("b", &[3])
                .unwrap_err()
                .to_string()
                .contains("expected one of [Bf16]")
        );
        assert!(
            st.f32("a", &[4])
                .unwrap_err()
                .to_string()
                .contains("shape [2, 2]")
        );
        assert!(
            st.f32("c", &[1])
                .unwrap_err()
                .to_string()
                .contains("no tensor named c")
        );
        let mut small = [0u8; 7];
        assert!(
            st.read_into("a", &mut small)
                .unwrap_err()
                .to_string()
                .contains("destination")
        );
    }

    #[test]
    fn refuses_malformed_headers() {
        let cases: [(&str, usize, &str); 7] = [
            ("not json", 0, "not JSON"),
            (r#"[1]"#, 0, "not a JSON object"),
            (
                r#"{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
                1,
                "dtype I8",
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                8,
                "span 4",
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                4,
                "do not fit",
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                    "b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
                8,
                "overlap",
            ),
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
                4,
                "not 2",
            ),
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
        let path =
            std::env::temp_dir().join(format!("qd-metal-st-short-{}-{n}", std::process::id()));
        std::fs::write(&path, [1u8, 2, 3]).unwrap();
        let f = TempFile(path);
        assert!(
            SafeTensors::open(&f.0)
                .err()
                .unwrap()
                .to_string()
                .contains("too short")
        );

        let path =
            std::env::temp_dir().join(format!("qd-metal-st-long-{}-{n}", std::process::id()));
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        std::fs::write(&path, bytes).unwrap();
        let f = TempFile(path);
        assert!(
            SafeTensors::open(&f.0)
                .err()
                .unwrap()
                .to_string()
                .contains("runs past")
        );
    }
}
