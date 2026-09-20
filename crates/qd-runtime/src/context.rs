//! `Context` — the one type in this crate that is deliberately *not* a `String`.
//!
//! `docs/schema-api.md`: "`context` crosses the FFI boundary as **bytes and a length, never a
//! Swift `String`**. A `String` round-trip normalizes Unicode and would silently change token
//! boundaries and therefore line spans."
//!
//! The enforcement is structural, not a comment:
//!
//! * there is no `From<String>`, no `From<&str>`, no `Deref<Target = str>`, no `Display`, and no
//!   `as_str()` — a caller cannot obtain a `Context` from a `String` through this crate's API, and
//!   cannot turn one back into a `String` and re-encode it;
//! * `Debug` prints the length and a digest, never the bytes, so a debug log cannot become the
//!   lossy round-trip the type exists to prevent;
//! * nothing here transcodes, normalizes, rewrites newlines, or validates UTF-8.
//!
//! # The wire form: `context_b64` + `context_len`, and nothing else
//!
//! `docs/schema-api.md`, *"The wire encoding of `context` — exactly one form"*:
//!
//! > **`context_b64` (base64, ASCII) plus a required `context_len` (the decoded byte count). They
//! > must agree, or it is a typed refusal.** There is no second accepted form.
//!
//! Base64 does not violate "never a `String`": the rule exists because a *Unicode* round-trip
//! rewrites token boundaries and line spans, and base64 is ASCII-only and byte-exact, so it does
//! not have the property the rule was written to prevent. `context_len` is kept even though base64
//! makes it derivable, because a disagreement between the two is the signature of a **truncated
//! payload that still decodes cleanly** — the one failure a length field is for.
//!
//! This module is the single owner of that encoding. [`Context::from_wire_parts`] is the only
//! decoder; [`crate::wire`] (which knows the JSON shape) and this type's own `Deserialize` both go
//! through it, so there is no second reading of the same two fields waiting to drift.
//!
//! Until 2026-09-19 this crate took `Context` as `#[serde(transparent)] Vec<u8>` — a JSON array of
//! integers — while `python/qd_data/schema.py` emitted `context_b64`. No caller could satisfy both
//! and the system did not work end to end, even though both sides' own suites passed. That was
//! `GAP-RT-WIRE-CONTEXT-ENCODING`; `tests/wire_context_crosslang.rs` is the standing check that
//! replaces the assumption the two lanes were each making about the other.

use serde::de::Error as DeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::refusal::Refusal;

/// The base64 field name on the wire. Matches `Request.to_wire()` in `python/qd_data/schema.py`.
pub const B64_FIELD: &str = "context_b64";
/// The decoded-byte-count field name on the wire.
pub const LEN_FIELD: &str = "context_len";

/// Raw context bytes, exactly as the caller measured them.
#[derive(Clone, PartialEq, Eq)]
pub struct Context {
    bytes: Vec<u8>,
}

impl Context {
    /// The only constructor from bytes. Takes ownership; performs no transcoding, no
    /// normalization, no newline rewriting, and no UTF-8 validity check.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Decode the one accepted wire form.
    ///
    /// Both parts are required. `None` for either is a typed refusal naming which one, rather than
    /// a default: a context that arrived without its length is a context whose truncation nobody
    /// would notice.
    ///
    /// The callers pass `Option` rather than the values because "absent" and "present but wrong"
    /// are different refusals, and both callers — [`crate::wire::validate`] and this type's
    /// `Deserialize` — have to produce the same one.
    ///
    /// A `declared_len` of `None` means the field was absent. A field that is *present* but is not
    /// a non-negative integer is the same refusal with a different `found` — operationally one
    /// condition, no usable declared length arrived — and the caller that saw the JSON raises it,
    /// because only the caller can name the JSON type. `python/qd_data/errors.py`'s
    /// `ContextLenMissingRefusal` covers the same two cases under the same identifier.
    pub fn from_wire_parts(b64: Option<&str>, declared_len: Option<u64>) -> Result<Self, Refusal> {
        let Some(b64) = b64 else {
            return Err(Refusal::ContextNotBytes {
                got: "absent".to_string(),
            });
        };
        let Some(declared_len) = declared_len else {
            return Err(Refusal::ContextLenMissing {
                found: "absent".to_string(),
            });
        };
        let bytes = crate::b64::decode(b64).map_err(|e| Refusal::ContextNotBase64 {
            offset: e.offset,
            expected: e.expected,
            found: e.found,
        })?;
        // Compared in `u64`, not in `usize`. A declared length larger than this platform can
        // address must refuse, and it must refuse for *disagreeing with the payload* — narrowing
        // first would make the comparison depend on the pointer width. `ContextLengthMismatch`
        // carries `usize`, so a value that cannot be represented is reported saturated; on every
        // platform this crate targets `usize` is 64 bits and the conversion is exact.
        if declared_len != bytes.len() as u64 {
            return Err(Refusal::ContextLengthMismatch {
                declared: usize::try_from(declared_len).unwrap_or(usize::MAX),
                actual: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    /// The bytes as canonical standard base64 — the `context_b64` field's value.
    pub fn to_b64(&self) -> String {
        crate::b64::encode(&self.bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Length in **bytes**. Never a char count, never a UTF-16 unit count. This is also the value
    /// `context_len` must carry.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// How much non-whitespace content the context carries. `docs/hardening.md` §3 requires an
    /// all-whitespace or empty context to refuse rather than produce a confident letter; this is
    /// what that check counts.
    ///
    /// Whitespace is the **Unicode** notion when the bytes are valid UTF-8, which is what
    /// `python/qd_data/render.py` uses (`str.strip()`), so a context of nothing but U+00A0 is
    /// refused by both lanes rather than by one. When the bytes are not valid UTF-8 there is no
    /// Unicode reading to take, so it falls back to counting non-ASCII-whitespace bytes; a caller
    /// on that path is heading for [`crate::refusal::Refusal::ContextNotUtf8`] anyway, and this
    /// keeps the function total instead of making it fallible for a case that cannot decide
    /// anything.
    pub fn non_whitespace_len(&self) -> usize {
        match std::str::from_utf8(&self.bytes) {
            Ok(s) => s.chars().filter(|c| !c.is_whitespace()).count(),
            Err(_) => self
                .bytes
                .iter()
                .filter(|b| !b.is_ascii_whitespace())
                .count(),
        }
    }

    /// Byte offsets of every line start: 0-based offsets, for 1-based line numbers.
    ///
    /// A `span` slot's pointer head ranges over exactly these positions, so this is the definition
    /// of "line" for the whole system. `\n` terminates a line; a trailing `\n` does not open an
    /// empty final line. `\r` is an ordinary content byte — `docs/hardening.md` §1 records CRLF as
    /// where byte offsets and line numbers diverge, so CR is never allowed to silently become a
    /// second terminator here.
    pub fn line_starts(&self) -> Vec<usize> {
        let mut starts = vec![0usize];
        for (i, b) in self.bytes.iter().enumerate() {
            if *b == b'\n' && i + 1 < self.bytes.len() {
                starts.push(i + 1);
            }
        }
        starts
    }

    /// Number of lines a `span` answer may point at.
    pub fn line_count(&self) -> usize {
        if self.bytes.is_empty() {
            0
        } else {
            self.line_starts().len()
        }
    }

    /// SHA-256 over the raw bytes. Seeds the request digest that drives the permutation, and backs
    /// the `Debug` rendering.
    pub fn digest(&self) -> [u8; 32] {
        crate::sha256(&self.bytes)
    }

    pub fn digest_hex(&self) -> String {
        crate::hex(&self.digest())
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.digest_hex();
        write!(
            f,
            "Context {{ len: {}, sha256: {} }}",
            self.bytes.len(),
            &d[..16]
        )
    }
}

/// Emits the two fields the contract names, and only those.
///
/// A `Context` nested in some other structure therefore carries the *same* encoding as one spliced
/// into a request envelope. That is the point: a second serialization of this type is how the
/// divergence this module documents would come back.
impl Serialize for Context {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Context", 2)?;
        s.serialize_field(B64_FIELD, &self.to_b64())?;
        s.serialize_field(LEN_FIELD, &(self.bytes.len() as u64))?;
        s.end()
    }
}

/// The struct form, used only to let serde do the shape work before
/// [`Context::from_wire_parts`] does the meaning work.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParts {
    #[serde(default)]
    context_b64: Option<String>,
    #[serde(default)]
    context_len: Option<u64>,
}

impl<'de> Deserialize<'de> for Context {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let parts = WireParts::deserialize(deserializer)?;
        // The refusal's `Display` already names the check and both values compared, so the serde
        // error a caller sees is as specific as the typed one `crate::wire` would have raised.
        Context::from_wire_parts(parts.context_b64.as_deref(), parts.context_len)
            .map_err(|refusal| D::Error::custom(refusal.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_serialized_form_is_exactly_the_two_contract_fields() {
        let ctx = Context::from_bytes(b"fn add() {}\n".to_vec());
        let value = serde_json::to_value(&ctx).expect("a context serializes");
        let object = value.as_object().expect("an object, not a bare array");
        assert_eq!(
            object.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![B64_FIELD, LEN_FIELD],
            "the wire form is these two fields and nothing else"
        );
        // A literal golden rather than a comparison against the function under test.
        assert_eq!(object[B64_FIELD], serde_json::json!("Zm4gYWRkKCkge30K"));
        assert_eq!(object[LEN_FIELD], serde_json::json!(12));
    }

    #[test]
    fn serialize_then_deserialize_is_the_identity() {
        for bytes in [
            Vec::new(),
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            vec![0x00, 0xff, 0xfe, 0x0b, 0x85],
            (0..=255u8).collect(),
        ] {
            let ctx = Context::from_bytes(bytes.clone());
            let json = serde_json::to_string(&ctx).expect("serializes");
            let back: Context = serde_json::from_str(&json).expect("round-trips");
            assert_eq!(back.as_bytes(), &bytes[..]);
            assert_eq!(back, ctx);
        }
    }

    #[test]
    fn a_declared_length_that_disagrees_is_refused_by_the_one_decoder() {
        let err = Context::from_wire_parts(Some("Zm9v"), Some(99)).expect_err("3 != 99");
        assert_eq!(
            err,
            Refusal::ContextLengthMismatch {
                declared: 99,
                actual: 3,
            }
        );
        // Both numbers reach the message a human reads.
        let message = err.to_string();
        assert!(message.contains("99") && message.contains('3'), "{message}");
    }

    #[test]
    fn each_missing_part_names_itself() {
        assert_eq!(
            Context::from_wire_parts(None, Some(0)),
            Err(Refusal::ContextNotBytes {
                got: "absent".to_string()
            })
        );
        assert_eq!(
            Context::from_wire_parts(Some(""), None),
            Err(Refusal::ContextLenMissing {
                found: "absent".to_string()
            })
        );
    }

    #[test]
    fn invalid_base64_is_a_typed_refusal_that_names_the_offset() {
        match Context::from_wire_parts(Some("Zm9v*g=="), Some(4)) {
            Err(Refusal::ContextNotBase64 {
                offset,
                expected,
                found,
            }) => {
                assert_eq!(offset, 4);
                assert!(expected.contains("alphabet"), "{expected}");
                assert_eq!(found, "`*`");
            }
            other => panic!("expected ContextNotBase64, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_context_is_a_legal_value_not_an_error() {
        let ctx = Context::from_wire_parts(Some(""), Some(0)).expect("the empty context decodes");
        assert!(ctx.is_empty());
        assert_eq!(ctx.line_count(), 0);
        assert_eq!(ctx.to_b64(), "");
    }

    #[test]
    fn debug_prints_a_length_and_a_digest_and_never_the_bytes() {
        let ctx = Context::from_bytes(b"SECRET-CONTEXT".to_vec());
        let rendered = format!("{ctx:?}");
        assert!(!rendered.contains("SECRET"), "{rendered}");
        assert!(rendered.contains("len: 14"), "{rendered}");
        assert!(rendered.contains(&ctx.digest_hex()[..16]), "{rendered}");
    }
}
