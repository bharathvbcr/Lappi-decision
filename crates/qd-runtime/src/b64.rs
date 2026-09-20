//! Base64 — RFC 4648 §4, standard alphabet, canonical padding.
//!
//! `docs/schema-api.md`, *"The wire encoding of `context` — exactly one form"*: the context crosses
//! the wire as `context_b64` plus a required `context_len`. This module is the codec, and
//! [`crate::context::Context`] is its only caller inside this crate.
//!
//! # The transform is the [`base64`] crate; the strictness is stated here
//!
//! This module was hand-written until 2026-09-19 under `CLAUDE.md`'s *"no new dependency without
//! asking"*. Asked and granted, so the transform is now the crate's — an audited, widely-used
//! implementation in place of ~120 lines of bit-shuffling on the request path.
//!
//! What the crate supplies is exactly the two configuration defaults this wire format depends on,
//! read out of the vendored source rather than recalled:
//!
//! ```text
//! GeneralPurposeConfig::new() -> decode_allow_trailing_bits: false
//!                                decode_padding_mode: RequireCanonical
//! STANDARD = GeneralPurpose::new(&alphabet::STANDARD, PAD)   // PAD == new()
//! ```
//!
//! **Those are config defaults, not guarantees of the format**, and a config default is precisely
//! what a version bump is free to re-tune. So the strictness stays a property this crate *asserts*
//! rather than inherits: [`tests::the_engine_config_this_module_depends_on`] pins both behaviours
//! directly, and a bump that loosened either fails a test instead of silently widening what the
//! runtime accepts.
//!
//! # Strictness, exactly
//!
//! [`decode`] accepts **only** canonical standard base64:
//!
//! * length a multiple of 4; no whitespace, no newlines, no URL-safe `-`/`_`, no missing padding;
//! * `=` only in the final quantum, and only as `xx==` or `xxx=`;
//! * the bits a partial quantum does not use must be **zero**.
//!
//! The last rule is the one worth naming: `"Zg=="` and `"Zh=="` both carry `f` in their data bits,
//! so a permissive decoder returns `b"f"` for both and two distinct strings name one context.
//! Python's `b64decode(validate=True)` is permissive there — it checks the alphabet, not the slack
//! bits — so `python/qd_data/schema.py` re-encodes and compares to close the same hole from its
//! side. The two lanes are strict in the same place on purpose, and
//! `tests/wire_context_crosslang.rs` asserts it rather than trusting it.
//!
//! # What this module still owns: the error
//!
//! [`crate::refusal::Refusal::ContextNotBase64`] carries an offset and both compared values, and
//! the crate's `DecodeError` is coarser than that — most importantly `InvalidPadding` carries no
//! offset at all. So the mapping below is not decoration; it is the difference between a refusal a
//! caller can act on and one that says "bad base64". The mapping was written from *measured* crate
//! behaviour, not from its docs — in particular, misplaced padding arrives as `InvalidByte` whose
//! byte is `=`, which is what lets a padding fault keep its own message.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

/// Why a base64 string could not be decoded. Carries the offset and both values the check
/// compared, so [`crate::refusal::Refusal::ContextNotBase64`] can name them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    /// Byte offset into the input at which the check failed.
    pub offset: usize,
    /// What a canonical encoding would have had there.
    pub expected: String,
    /// What was there instead.
    pub found: String,
}

/// Sextet -> character, for naming the canonical form of a symbol in an error message. Not a
/// codec: nothing here decodes, and the crate owns the transform.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const PAD: u8 = b'=';

/// Encode bytes as canonical standard base64 with padding.
pub fn encode(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

/// Decode canonical standard base64. See the module docs for exactly what is accepted.
pub fn decode(text: &str) -> Result<Vec<u8>, DecodeError> {
    let src = text.as_bytes();

    // Checked here rather than mapped from the crate's answer. A non-multiple-of-4 input surfaces
    // as `InvalidPadding`, which carries no offset and no length, so the crate cannot say how far
    // off the input was — and "the nearest is 4" is the one thing a caller with a truncated payload
    // needs. Cheap, total, and it keeps the most likely fault the most legible.
    if !src.len().is_multiple_of(4) {
        return Err(DecodeError {
            offset: src.len(),
            expected: format!(
                "a length that is a multiple of 4 (canonical padding); the nearest is {}",
                src.len().next_multiple_of(4)
            ),
            found: format!("{} characters", src.len()),
        });
    }

    STANDARD.decode(text).map_err(|e| map_error(e, text))
}

fn map_error(err: base64::DecodeError, text: &str) -> DecodeError {
    let src = text.as_bytes();
    match err {
        // Measured: misplaced padding arrives here with `byte == b'='`, not as `InvalidPadding`.
        // `"=Zm9"`, `"Z==="`, `"Zm=v"` and `"Zg==Zg=="` all take this arm. Splitting on the byte is
        // what keeps a padding fault from being reported as an alphabet fault.
        base64::DecodeError::InvalidByte(offset, byte) if byte == PAD => DecodeError {
            offset,
            expected: "a base64 character; `=` is legal only as the last one or two characters \
                       of the whole string"
                .to_string(),
            found: "`=`".to_string(),
        },
        base64::DecodeError::InvalidByte(offset, byte) => DecodeError {
            offset,
            expected: "a character from the standard base64 alphabet (A-Z a-z 0-9 + /) or `=`"
                .to_string(),
            found: describe(byte, text, offset),
        },
        // The slack-bits fault: a symbol in the alphabet whose low bits are not part of any byte.
        // `symbol_value` is the sextet the symbol denotes, so the canonical symbol is that value
        // with the unused low bits cleared.
        base64::DecodeError::InvalidLastSymbol {
            offset,
            symbol,
            symbol_value,
        } => {
            // Symbols in the final quantum up to and including this one. `offset` is the last
            // non-padding symbol, so this is 2 for `xx==` and 3 for `xxx=`.
            let quantum_start = (src.len().saturating_sub(1) / 4) * 4;
            let kept = offset.saturating_sub(quantum_start) + 1;
            let slack_bits = 8usize.saturating_sub(2 * kept);
            let within_sextet = if slack_bits >= 8 { 0xffu8 } else { (1u8 << slack_bits) - 1 };
            let canonical = symbol_value & !within_sextet & 0x3f;
            DecodeError {
                offset,
                expected: format!(
                    "zero in the {slack_bits} unused low bits of the final quantum; `{}` is the \
                     canonical character for these data bits",
                    ALPHABET[canonical as usize] as char
                ),
                found: format!(
                    "`{}`, whose low {slack_bits} bits are {:#x}",
                    symbol as char,
                    symbol_value & within_sextet
                ),
            }
        }
        base64::DecodeError::InvalidLength(len) => DecodeError {
            offset: len,
            expected: format!(
                "a length that is a multiple of 4 (canonical padding); the nearest is {}",
                len.next_multiple_of(4)
            ),
            found: format!("{len} characters"),
        },
        // The one variant the crate gives no offset for. The length pre-check above already
        // rejects the common cause, so reaching here means padding that is wrong in some other
        // way; the first `=` is the only defensible location to name, and saying it is synthesised
        // beats reporting offset 0 as though it had been measured.
        base64::DecodeError::InvalidPadding => {
            let offset = src.iter().position(|c| *c == PAD).unwrap_or(src.len());
            DecodeError {
                offset,
                expected: "canonical padding: `=` only as the last one or two characters, and \
                           present whenever the final quantum is short"
                    .to_string(),
                found: "padding that is absent, misplaced, or the wrong length (the decoder \
                        reports this fault without an offset; the position named is the first \
                        `=`)"
                    .to_string(),
            }
        }
    }
}

/// Name an offending character without reproducing an arbitrary blob in an error message. A
/// non-ASCII byte is named by its code point rather than printed raw, because the byte alone is not
/// a character.
fn describe(raw: u8, text: &str, offset: usize) -> String {
    if raw.is_ascii_graphic() {
        return format!("`{}`", raw as char);
    }
    match text.get(offset..).and_then(|rest| rest.chars().next()) {
        Some(c) => format!("U+{:04X}", c as u32),
        None => format!("byte {raw:#04x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two `base64` engine defaults this module's contract rests on.
    ///
    /// `STANDARD` is strict here only because `GeneralPurposeConfig::new()` sets
    /// `decode_allow_trailing_bits: false` and `decode_padding_mode: RequireCanonical`. Those are
    /// *defaults*, and a version bump may change a default without changing an API. This test
    /// pins the behaviour directly against the engine, so such a bump fails here rather than
    /// quietly widening what the runtime accepts off the wire.
    #[test]
    fn the_engine_config_this_module_depends_on() {
        // decode_allow_trailing_bits == false
        assert!(
            STANDARD.decode("Zh==").is_err(),
            "engine now accepts non-zero trailing bits; `Zg==` and `Zh==` would both name b\"f\""
        );
        // decode_padding_mode == RequireCanonical
        assert!(
            STANDARD.decode("Zm9").is_err(),
            "engine now accepts missing padding; two strings would name one context"
        );
        assert_eq!(STANDARD.decode("Zg==").as_deref(), Ok(b"f".as_slice()));
    }

    /// RFC 4648 §10, verbatim.
    #[test]
    fn rfc4648_test_vectors() {
        let vectors: &[(&str, &str)] = &[
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (plain, encoded) in vectors {
            assert_eq!(&encode(plain.as_bytes()), encoded, "encode({plain:?})");
            assert_eq!(
                decode(encoded).as_deref(),
                Ok(plain.as_bytes()),
                "decode({encoded:?})"
            );
        }
    }

    #[test]
    fn every_byte_value_round_trips_at_every_alignment() {
        for len in 0..=8usize {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(decode(&encode(&bytes)), Ok(bytes.clone()), "len {len}");
        }
        let all: Vec<u8> = (0..=255u8).collect();
        assert_eq!(decode(&encode(&all)), Ok(all));
        // The 0xFF-heavy case: every sextet is 63, so the final quantum's slack bits matter.
        for len in 1..=6usize {
            let bytes = vec![0xffu8; len];
            assert_eq!(decode(&encode(&bytes)), Ok(bytes.clone()), "0xff x {len}");
        }
    }

    #[test]
    fn a_non_multiple_of_four_is_refused() {
        let err = decode("Zm9").expect_err("three characters is not a whole quantum");
        assert!(err.expected.contains("multiple of 4"), "{err:?}");
        assert_eq!(err.found, "3 characters");
    }

    #[test]
    fn characters_outside_the_alphabet_are_refused() {
        // Whitespace, newlines and the URL-safe alphabet are all outside standard base64.
        for bad in ["Zm 8=", "Zm\n8=", "Zm9-", "Zm9_", "\u{00e9}\u{00e9}"] {
            assert!(decode(bad).is_err(), "{bad:?} must not decode");
        }
    }

    #[test]
    fn non_canonical_padding_is_refused() {
        for bad in ["=Zm9", "Z===", "====", "Zm=v", "Zg==Zg=="] {
            assert!(decode(bad).is_err(), "{bad:?} must not decode");
        }
    }

    /// A padding fault must not be reported as an alphabet fault. The crate delivers both as
    /// `InvalidByte`, so this is the mapping's own correctness, not the crate's.
    #[test]
    fn a_misplaced_pad_is_named_as_padding_not_as_a_bad_character() {
        let err = decode("Zm=v").expect_err("`=` mid-quantum is not canonical");
        assert_eq!(err.offset, 2);
        assert_eq!(err.found, "`=`");
        assert!(err.expected.contains("legal only as the last"), "{err:?}");
    }

    #[test]
    fn non_zero_trailing_bits_are_refused() {
        // `Zg==` and `Zh==` both carry `f` in their data bits; a permissive decoder returns b"f"
        // for both, so two strings would name one context.
        assert_eq!(decode("Zg=="), Ok(b"f".to_vec()));
        let err = decode("Zh==").expect_err("non-zero slack bits must not decode");
        assert!(err.expected.contains("unused low bits"), "{err:?}");
        assert!(err.expected.contains("`g`"), "names the canonical form: {err:?}");
        // The same at the three-character boundary.
        assert_eq!(decode("Zm8="), Ok(b"fo".to_vec()));
        assert!(decode("Zm9=").is_err());
    }

    /// The slack-bits error must name the canonical character at both quantum widths, since the
    /// mapping derives the unused-bit count from the offset rather than being told it.
    #[test]
    fn the_slack_bits_error_names_the_canonical_character_at_both_widths() {
        let two = decode("Zh==").expect_err("xx== with slack bits");
        assert_eq!(two.offset, 1);
        assert!(two.expected.contains("4 unused low bits"), "{two:?}");
        assert!(two.expected.contains("`g`"), "{two:?}");

        let three = decode("Zm9=").expect_err("xxx= with slack bits");
        assert_eq!(three.offset, 2);
        assert!(three.expected.contains("2 unused low bits"), "{three:?}");
        assert!(three.expected.contains("`8`"), "canonical for `9` is `8`: {three:?}");
    }

    #[test]
    fn an_error_names_the_offset_and_both_values() {
        let err = decode("Zm9v*g==").expect_err("`*` is not in the alphabet");
        assert_eq!(err.offset, 4);
        assert!(err.expected.contains("standard base64 alphabet"), "{err:?}");
        assert_eq!(err.found, "`*`");
    }
}
