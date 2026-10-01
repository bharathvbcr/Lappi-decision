//! A hand-rolled reader for NumPy's `.npy` format, versions 1.0-3.0 (no dependency).
//!
//! The shard set stores `offsets.npy` with `np.save` and five arrays inside `supervision.npz`
//! (each member is itself an `.npy`). The format is a magic string, a version, a header
//! length, a Python dict literal (`{'descr': '<i8', 'fortran_order': False, 'shape': (99,), }`)
//! and the raw C-order data. Everything this crate does not need is refused by name rather
//! than half-supported: big-endian data, Fortran order, structured or object dtypes, and any
//! byte count that disagrees with `shape x itemsize`.

use thiserror::Error;

const MAGIC: &[u8] = b"\x93NUMPY";
/// Ceiling on the header dict's length; numpy's own limit for v1 is 65,535 bytes.
const MAX_HEADER_LEN: usize = 1 << 20;
/// Ceiling on dimensions; the shard contract never stores more than two.
const MAX_NDIM: usize = 8;

/// The element types this reader accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// `|b1`.
    Bool,
    /// `|u1`.
    U8,
    /// `|i1`.
    I8,
    /// `<u2`.
    U16,
    /// `<i2`.
    I16,
    /// `<u4`.
    U32,
    /// `<i4`.
    I32,
    /// `<u8`.
    U64,
    /// `<i8`.
    I64,
}

impl Dtype {
    /// Bytes per element.
    pub fn itemsize(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 => 2,
            Dtype::U32 | Dtype::I32 => 4,
            Dtype::U64 | Dtype::I64 => 8,
        }
    }

    fn parse(descr: &str) -> Result<Self, NpyError> {
        let (Some(order), Some(kind)) = (descr.get(..1), descr.get(1..)) else {
            return Err(NpyError::Unsupported(format!("dtype {descr:?}")));
        };
        let single_byte = matches!(kind, "b1" | "u1" | "i1");
        match order {
            "<" | "=" => {}
            "|" if single_byte => {}
            ">" if single_byte => {}
            ">" => {
                return Err(NpyError::Unsupported(format!(
                    "big-endian dtype {descr:?}; every array this crate reads is little-endian"
                )));
            }
            _ => return Err(NpyError::Unsupported(format!("dtype {descr:?}"))),
        }
        Ok(match kind {
            "b1" => Dtype::Bool,
            "u1" => Dtype::U8,
            "i1" => Dtype::I8,
            "u2" => Dtype::U16,
            "i2" => Dtype::I16,
            "u4" => Dtype::U32,
            "i4" => Dtype::I32,
            "u8" => Dtype::U64,
            "i8" => Dtype::I64,
            _ => return Err(NpyError::Unsupported(format!("dtype {descr:?}"))),
        })
    }
}

/// An `.npy` this reader refuses.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NpyError {
    /// Not an `.npy`, or truncated, or internally inconsistent.
    #[error("malformed .npy: {0}")]
    Malformed(String),
    /// A valid `.npy` using a feature this reader does not support.
    #[error("unsupported .npy: {0}")]
    Unsupported(String),
    /// An element that does not fit the type the caller asked for.
    #[error("{0}")]
    OutOfRange(String),
}

/// One array: its dtype, its C-order shape and its raw little-endian bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpyArray {
    /// Element type.
    pub dtype: Dtype,
    /// Shape, C order.
    pub shape: Vec<usize>,
    data: Vec<u8>,
}

impl NpyArray {
    /// Parse a whole `.npy` file's bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, NpyError> {
        if bytes.len() < MAGIC.len() + 2 || &bytes[..MAGIC.len()] != MAGIC {
            return Err(NpyError::Malformed(
                "missing the \\x93NUMPY magic".to_owned(),
            ));
        }
        let major = bytes[MAGIC.len()];
        let (header_len, header_start) = match major {
            1 => {
                let at = MAGIC.len() + 2;
                let raw = bytes
                    .get(at..at + 2)
                    .ok_or_else(|| NpyError::Malformed("truncated header length".to_owned()))?;
                (usize::from(u16::from_le_bytes([raw[0], raw[1]])), at + 2)
            }
            2 | 3 => {
                let at = MAGIC.len() + 2;
                let raw = bytes
                    .get(at..at + 4)
                    .ok_or_else(|| NpyError::Malformed("truncated header length".to_owned()))?;
                let len = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                (usize::try_from(len).unwrap_or(usize::MAX), at + 4)
            }
            v => return Err(NpyError::Unsupported(format!("format version {v}"))),
        };
        if header_len > MAX_HEADER_LEN {
            return Err(NpyError::Malformed(format!("header of {header_len} bytes")));
        }
        let header_end = header_start
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                NpyError::Malformed("header runs past the end of the file".to_owned())
            })?;
        let header = std::str::from_utf8(&bytes[header_start..header_end])
            .map_err(|_| NpyError::Malformed("header is not UTF-8".to_owned()))?;
        let dict = parse_header_dict(header)?;
        let dtype = Dtype::parse(&dict.descr)?;
        if dict.fortran_order {
            return Err(NpyError::Unsupported(
                "fortran_order=True; the shard contract stores C-order arrays".to_owned(),
            ));
        }
        let count = dict
            .shape
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d))
            .ok_or_else(|| NpyError::Malformed(format!("shape {:?} overflows", dict.shape)))?;
        let expected = count
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| NpyError::Malformed(format!("shape {:?} overflows", dict.shape)))?;
        let data = &bytes[header_end..];
        if data.len() != expected {
            return Err(NpyError::Malformed(format!(
                "{} data bytes for shape {:?} of {:?} ({expected} expected)",
                data.len(),
                dict.shape,
                dtype
            )));
        }
        Ok(Self {
            dtype,
            shape: dict.shape,
            data: data.to_vec(),
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len() / self.dtype.itemsize()
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Every element as `i128`, which holds every supported integer type exactly.
    fn values(&self) -> Vec<i128> {
        let size = self.dtype.itemsize();
        self.data
            .chunks_exact(size)
            .map(|c| match self.dtype {
                Dtype::Bool | Dtype::U8 => i128::from(c[0]),
                Dtype::I8 => i128::from(i8::from_le_bytes([c[0]])),
                Dtype::U16 => i128::from(u16::from_le_bytes([c[0], c[1]])),
                Dtype::I16 => i128::from(i16::from_le_bytes([c[0], c[1]])),
                Dtype::U32 => i128::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                Dtype::I32 => i128::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                Dtype::U64 => i128::from(u64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ])),
                Dtype::I64 => i128::from(i64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ])),
            })
            .collect()
    }

    /// The elements of an exactly-`int64` array. `offsets.npy` must be this dtype; Python's
    /// reader refuses anything else rather than casting.
    pub fn exact_i64(&self) -> Result<Vec<i64>, NpyError> {
        if self.dtype != Dtype::I64 {
            return Err(NpyError::Unsupported(format!(
                "dtype {:?}, expected int64",
                self.dtype
            )));
        }
        Ok(self
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    }

    /// Any integer array as `T`, refusing an element that does not fit rather than wrapping.
    ///
    /// Python's reader writes `z[name].astype(np.int32)` and friends, which wrap silently on an
    /// out-of-range element; the writer never produces one, so refusing is the same answer on
    /// every set Python writes and a loud one on any set it does not.
    pub fn cast<T: TryFrom<i128>>(&self, what: &str) -> Result<Vec<T>, NpyError> {
        if self.dtype == Dtype::Bool {
            return Err(NpyError::Unsupported(format!(
                "{what}: bool array where integers belong"
            )));
        }
        self.values()
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                T::try_from(v).map_err(|_| {
                    NpyError::OutOfRange(format!(
                        "{what}[{i}] = {v} does not fit {}",
                        std::any::type_name::<T>()
                    ))
                })
            })
            .collect()
    }
}

struct HeaderDict {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// The header dict literal: exactly the keys `descr`, `fortran_order`, `shape`.
fn parse_header_dict(header: &str) -> Result<HeaderDict, NpyError> {
    let mut p = Lit {
        s: header.as_bytes(),
        i: 0,
    };
    p.ws();
    p.expect(b'{')?;
    let mut descr: Option<String> = None;
    let mut fortran: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;
    loop {
        p.ws();
        if p.eat(b'}') {
            break;
        }
        let key = p.string()?;
        p.ws();
        p.expect(b':')?;
        p.ws();
        match key.as_str() {
            "descr" => descr = Some(p.string()?),
            "fortran_order" => fortran = Some(p.boolean()?),
            "shape" => shape = Some(p.tuple()?),
            other => {
                return Err(NpyError::Unsupported(format!("header key {other:?}")));
            }
        }
        p.ws();
        if p.eat(b',') {
            continue;
        }
        p.ws();
        p.expect(b'}')?;
        break;
    }
    p.ws();
    if p.i != p.s.len() {
        return Err(NpyError::Malformed(
            "trailing bytes after the header dict".to_owned(),
        ));
    }
    let shape = shape.ok_or_else(|| NpyError::Malformed("header has no 'shape'".to_owned()))?;
    if shape.len() > MAX_NDIM {
        return Err(NpyError::Unsupported(format!("{} dimensions", shape.len())));
    }
    Ok(HeaderDict {
        descr: descr.ok_or_else(|| NpyError::Malformed("header has no 'descr'".to_owned()))?,
        fortran_order: fortran
            .ok_or_else(|| NpyError::Malformed("header has no 'fortran_order'".to_owned()))?,
        shape,
    })
}

struct Lit<'a> {
    s: &'a [u8],
    i: usize,
}

impl Lit<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), NpyError> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(NpyError::Malformed(format!(
                "expected {:?} at byte {} of the header",
                char::from(c),
                self.i
            )))
        }
    }

    fn string(&mut self) -> Result<String, NpyError> {
        let quote = match self.s.get(self.i) {
            Some(q @ (b'\'' | b'"')) => *q,
            _ => {
                return Err(NpyError::Malformed(format!(
                    "expected a string at byte {}",
                    self.i
                )));
            }
        };
        self.i += 1;
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i] != quote {
            if self.s[self.i] == b'\\' {
                return Err(NpyError::Unsupported(
                    "escapes in a header string".to_owned(),
                ));
            }
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return Err(NpyError::Malformed("unterminated header string".to_owned()));
        }
        let out = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        self.i += 1;
        Ok(out)
    }

    fn boolean(&mut self) -> Result<bool, NpyError> {
        if self.s[self.i..].starts_with(b"True") {
            self.i += 4;
            Ok(true)
        } else if self.s[self.i..].starts_with(b"False") {
            self.i += 5;
            Ok(false)
        } else {
            Err(NpyError::Malformed(format!(
                "expected True/False at byte {}",
                self.i
            )))
        }
    }

    fn tuple(&mut self) -> Result<Vec<usize>, NpyError> {
        self.expect(b'(')?;
        let mut dims = Vec::new();
        loop {
            self.ws();
            if self.eat(b')') {
                return Ok(dims);
            }
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            // numpy writes Python ints; an `L` suffix only ever came from Python 2.
            let digits = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
            let dim: usize = digits.parse().map_err(|_| {
                NpyError::Malformed(format!("expected a dimension at byte {start}"))
            })?;
            dims.push(dim);
            self.ws();
            if self.eat(b',') {
                continue;
            }
            self.ws();
            self.expect(b')')?;
            return Ok(dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_a_one_d_int64_array() {
        let mut data = Vec::new();
        for v in [0i64, 5, -7] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let a = NpyArray::parse(&npy(
            "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }      \n",
            &data,
        ))
        .unwrap();
        assert_eq!(a.shape, vec![3]);
        assert_eq!(a.exact_i64().unwrap(), vec![0, 5, -7]);
        assert_eq!(a.cast::<i32>("x").unwrap(), vec![0, 5, -7]);
        assert!(
            a.cast::<u8>("x").is_err(),
            "a negative int64 must not wrap into u8"
        );
    }

    #[test]
    fn refuses_fortran_order_big_endian_and_short_data() {
        let h = |d: &str, f: &str| {
            format!("{{'descr': '{d}', 'fortran_order': {f}, 'shape': (1,), }}\n")
        };
        assert!(NpyArray::parse(&npy(&h("<i4", "True"), &[0; 4])).is_err());
        assert!(NpyArray::parse(&npy(&h(">i4", "False"), &[0; 4])).is_err());
        assert!(NpyArray::parse(&npy(&h("<i4", "False"), &[0; 3])).is_err());
        assert!(NpyArray::parse(&npy(&h("<f4", "False"), &[0; 4])).is_err());
        assert!(NpyArray::parse(&npy(&h("|u1", "False"), &[7])).is_ok());
    }

    #[test]
    fn reads_two_d_and_empty_shapes() {
        let a = NpyArray::parse(&npy(
            "{'descr': '<i4', 'fortran_order': False, 'shape': (2, 2), }\n",
            &[0u8; 16],
        ))
        .unwrap();
        assert_eq!(a.shape, vec![2, 2]);
        let e = NpyArray::parse(&npy(
            "{'descr': '<i4', 'fortran_order': False, 'shape': (0,), }\n",
            &[],
        ))
        .unwrap();
        assert!(e.is_empty());
    }
}
