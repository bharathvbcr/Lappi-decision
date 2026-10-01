//! The one cast the export makes: a float tensor to bf16.
//!
//! **Rounding is to nearest, ties to even.** An f32 is `sign | 8-bit exponent | 23-bit
//! mantissa`; a bf16 is the top 16 bits of the same layout. Truncating would always round
//! toward zero and bias every weight's magnitude down. Instead the low 16 bits are rounded:
//! add `0x7FFF` plus the lowest *kept* bit, then shift. Above the halfway point that carries
//! into the kept bits (round up); below it does not (round down); exactly at halfway it
//! carries only when the kept lowest bit is 1, which leaves the result even. The carry can run
//! into the exponent, which is correct: it is the next representable value. It is the rule
//! torch's `.to(torch.bfloat16)` and tessl's `f32_to_bf16_bits` use; the tests check this
//! against both.
//!
//! **Only precision may be lost.** bf16 has f32's exponent range, so a finite f32 loses only
//! mantissa bits -- except within half an ulp of `f32::MAX`, where rounding up reaches
//! infinity. That value is refused, as is any NaN or infinity already in the source: a weight
//! that is not a finite number is a broken checkpoint, and the release would carry it into every
//! answer. F16 widens to f32 exactly (every half is an f32), so an F16 source is rounded once.

use crate::safetensors::Dtype;

/// The float dtypes a source tensor may hold: the three `load_text_tower` can build a tower in
/// (`python/qd_train/backbone.py:445-449`), which `tools/ckpt_average.py` casts the mean back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSource {
    Bf16,
    F16,
    F32,
}

impl FloatSource {
    pub fn from_dtype(d: Dtype) -> Option<Self> {
        match d {
            Dtype::Bf16 => Some(FloatSource::Bf16),
            Dtype::F16 => Some(FloatSource::F16),
            Dtype::F32 => Some(FloatSource::F32),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            FloatSource::Bf16 | FloatSource::F16 => 2,
            FloatSource::F32 => 4,
        }
    }
}

/// `x` as bf16 bits, rounded to nearest-even. `None` when `x` is NaN or infinite, or rounds to
/// infinity.
pub fn f32_to_bf16_rne(x: f32) -> Option<u16> {
    if !x.is_finite() {
        return None;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    // Finite f32 bits are at most 0xFF7F_FFFF, so adding at most 0x8000 cannot overflow u32.
    let rounded = bits.checked_add(0x7FFF + lsb)?;
    let out = u16::try_from(rounded >> 16).ok()?;
    if !bf16_is_finite(out) {
        return None;
    }
    Some(out)
}

pub fn bf16_is_finite(b: u16) -> bool {
    b & 0x7F80 != 0x7F80
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// IEEE binary16 bits -> f32, exactly. Subnormal halves become normal f32s; infinities and NaNs
/// keep their class (and are refused by the caller).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x03FF);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // mant * 2^-24 with mant in [1, 1023]: normalise by the leading set bit.
            let msb = 31 - mant.leading_zeros();
            let e = msb + 127 - 24;
            let frac = (mant << (23 - msb)) & 0x007F_FFFF;
            sign | (e << 23) | frac
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// A value the cast refuses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BadValue {
    /// Element index within the tensor.
    pub index: u64,
    /// The source value, widened to f32.
    pub value: f32,
}

/// Convert little-endian `src` elements to little-endian bf16, appending to `out`.
/// `first_index` is the tensor element index of `src[0]`, for the error. `src` holds whole
/// elements: the export reads tensors in chunks that are a multiple of every element size.
pub fn to_bf16(kind: FloatSource, src: &[u8], first_index: u64, out: &mut Vec<u8>) -> Result<(), BadValue> {
    match kind {
        FloatSource::F32 => {
            for (index, c) in (first_index..).zip(src.as_chunks::<4>().0) {
                let x = f32::from_le_bytes(*c);
                let b = f32_to_bf16_rne(x).ok_or(BadValue { index, value: x })?;
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        FloatSource::F16 => {
            for (index, c) in (first_index..).zip(src.as_chunks::<2>().0) {
                let x = f16_bits_to_f32(u16::from_le_bytes(*c));
                let b = f32_to_bf16_rne(x).ok_or(BadValue { index, value: x })?;
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
        FloatSource::Bf16 => {
            for (index, c) in (first_index..).zip(src.as_chunks::<2>().0) {
                let b = u16::from_le_bytes(*c);
                if !bf16_is_finite(b) {
                    return Err(BadValue { index, value: bf16_bits_to_f32(b) });
                }
            }
            out.extend_from_slice(src);
        }
    }
    Ok(())
}

/// Check that little-endian f32 `src` is all finite, for a tensor that is copied as is.
pub fn check_finite_f32(src: &[u8], first_index: u64) -> Result<(), BadValue> {
    for (index, c) in (first_index..).zip(src.as_chunks::<4>().0) {
        let x = f32::from_le_bytes(*c);
        if !x.is_finite() {
            return Err(BadValue { index, value: x });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_to_nearest_with_ties_to_even() {
        // 1.0 + 2^-8 is exactly halfway between bf16 1.0 (0x3F80) and the next value (0x3F81);
        // 0x3F80 is even, so the tie goes down.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8000)), Some(0x3F80));
        // Halfway above an odd kept value goes up to the even one.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F81_8000)), Some(0x3F82));
        // Just above / below halfway.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8001)), Some(0x3F81));
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_7FFF)), Some(0x3F80));
        // Carry into the exponent: the largest mantissa rounds to the next power of two.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3FFF_FFFF)), Some(0x4000));
        // Sign, zero and subnormals are kept.
        assert_eq!(f32_to_bf16_rne(-0.0), Some(0x8000));
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x0001_0000)), Some(0x0001));
        assert_eq!(f32_to_bf16_rne(-1.5), Some(0xBFC0));
        // Exactly representable values are unchanged.
        for v in [1.0f32, -2.0, 0.5, 3.0, 2f32.powi(-100), -7.25] {
            assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne(v).unwrap()), v);
        }
    }

    #[test]
    fn refuses_what_is_not_a_finite_bf16() {
        assert_eq!(f32_to_bf16_rne(f32::NAN), None);
        assert_eq!(f32_to_bf16_rne(f32::INFINITY), None);
        assert_eq!(f32_to_bf16_rne(f32::NEG_INFINITY), None);
        // f32::MAX is above bf16's largest finite value plus half an ulp: it rounds to inf.
        assert_eq!(f32_to_bf16_rne(f32::MAX), None);
        assert_eq!(f32_to_bf16_rne(-f32::MAX), None);
        // bf16's largest finite value survives.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x7F7F_0000)), Some(0x7F7F));
    }

    #[test]
    fn f16_widens_exactly() {
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xC000), -2.0);
        assert_eq!(f16_bits_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        assert!(f16_bits_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn chunk_conversion_reports_the_first_bad_element() {
        let mut src = Vec::new();
        for v in [1.0f32, 2.0, f32::NAN, 4.0] {
            src.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = Vec::new();
        let err = to_bf16(FloatSource::F32, &src, 100, &mut out).unwrap_err();
        assert_eq!(err.index, 102);
        assert!(err.value.is_nan());

        let bf: Vec<u8> = [0x3F80u16, 0x7F80].iter().flat_map(|b| b.to_le_bytes()).collect();
        assert_eq!(to_bf16(FloatSource::Bf16, &bf, 0, &mut Vec::new()).unwrap_err().index, 1);
        let ok: Vec<u8> = [0x3F80u16, 0xC000].iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut out = Vec::new();
        to_bf16(FloatSource::Bf16, &ok, 0, &mut out).unwrap();
        assert_eq!(out, ok, "a bf16 source is copied bit for bit");

        // f16 0x3555 = 0.33325195 = f32 0x3EAA_A000; the dropped 0xA000 is above halfway, so it
        // rounds up to 0x3EAB.
        let half: Vec<u8> = [0x3C00u16, 0x3555].iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut out = Vec::new();
        to_bf16(FloatSource::F16, &half, 0, &mut out).unwrap();
        assert_eq!(out, [0x3F80u16, 0x3EAB].iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());

        assert_eq!(check_finite_f32(&src, 0).unwrap_err().index, 2);
    }
}
