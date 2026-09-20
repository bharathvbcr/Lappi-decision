//! Prompt rendering. `docs/hardening.md` §3: **the context is adversarial by construction.**
//!
//! This module is the Rust half of a two-language contract. The other half is
//! `python/qd_data/render.py`, and the two must produce **byte-identical** prompts for the same
//! request — `docs/schema-api.md`: *"the training format and the serving format are the same
//! object. If they drift, the model is served a prompt shape it never saw, and nothing in the eval
//! catches it."* Every constant, every marker and the escape scanner below are transcriptions of
//! that file, kept in the same order so a diff of the two is readable.
//!
//! # The one invariant
//!
//! **No untrusted byte can produce the two-character sequence `<|` in the rendered prompt.**
//!
//! Every structural marker has the shape `<|qd_...|>`, and every special token in the Qwen
//! tokenizer has the shape `<|...|>`. One escape rule — break `<|` into `<\|` — therefore stops all
//! three attacks in the hardening table at once: forging a delimiter to terminate the context
//! early, forging an option line to create a 17th apparent option, and smuggling `<|im_start|>` in
//! as a control token rather than as text. It is one rule instead of a blocklist because a
//! blocklist goes stale the moment the tokenizer gains a token, and the failure is silent.
//!
//! Two further invariants, both asserted in `tests/render_contract.rs`:
//!
//! * [`escape_block`] preserves the newline count **exactly**. Span slots answer with line numbers
//!   into the context, so an escape that added or removed a line would move every span label.
//! * [`escape_inline`] emits no newline at all, which is what makes "one option, one line" true no
//!   matter what the option text contains.
//!
//! # What this module does not do
//!
//! It does not tokenize. The contract's requirement that the context region be tokenized with
//! special-token handling **disabled** cannot be discharged here, so it is carried in the type
//! instead: [`RenderedPrompt::context_region`] names the exact byte range a backend must encode as
//! ordinary text. A backend that ignores it is not caught by this crate; that is stated rather than
//! papered over.

use crate::refusal::Refusal;
use crate::schema::{check_slot_name, DecisionRequest, Route, SlotSpec, NOUL_LABEL};

// -- structural markers -------------------------------------------------------------------------
// Transcribed from python/qd_data/render.py. Each is of the form <|qd_...|>; see the module docs.

pub const M_BEGIN: &str = "<|qd_begin|>";
pub const M_VERSION: &str = "<|qd_schema_version|>";
pub const M_TASK: &str = "<|qd_task|>";
pub const M_ROUTE: &str = "<|qd_route|>";
pub const M_QUESTION: &str = "<|qd_question|>";
pub const M_CTX_BEGIN: &str = "<|qd_context_begin|>";
pub const M_CTX_END: &str = "<|qd_context_end|>";
pub const M_SLOT: &str = "<|qd_slot|>";
pub const M_TYPE: &str = "<|qd_type|>";
pub const M_OPT_BEGIN: &str = "<|qd_options_begin|>";
pub const M_OPT_END: &str = "<|qd_options_end|>";
pub const M_ANSWER: &str = "<|qd_answer|>";

pub const MARKERS: &[&str] = &[
    M_BEGIN,
    M_VERSION,
    M_TASK,
    M_ROUTE,
    M_QUESTION,
    M_CTX_BEGIN,
    M_CTX_END,
    M_SLOT,
    M_TYPE,
    M_OPT_BEGIN,
    M_OPT_END,
    M_ANSWER,
];

/// The 16 **named-option** letters the generic route decodes over: `docs/schema-api.md`, "one token
/// over a 17-row `lm_head` slice: 16 option letters + 1 reserved `noul` row". The seventeenth row's
/// letter is [`NOUL_LETTER`], outside this block.
pub const OPTION_LETTERS: &[u8; 16] = b"ABCDEFGHIJKLMNOP";

/// `noul` is *"a reserved letter present in every option set"*, given a letter **outside** the
/// contiguous A-P block so the 16-row option slice stays exactly 16 rows even when a request uses
/// all 16 named options.
///
/// `python/qd_data/schema.py` records this as the open question `GAP-SCHEMA-NOUL-LETTER-BUDGET`
/// and asks the runtime lane to settle it. **Settled here as 16 option rows plus one reserved
/// `noul` row** — 17 rows for a maximal choice slot. [`crate::schema::SlotSpec::answer_rows`]
/// already encoded that reading, and the alternative (15 named options) would make
/// `docs/schema-api.md`'s own "k <= 16" false. `MAX_CHOICE_OPTIONS` stays 16 on the Python side.
pub const NOUL_LETTER: u8 = b'Z';

// -- caps ---------------------------------------------------------------------------------------

/// Every payload is bounded, and the bounds are checked before the model sees anything.
/// `docs/schema-api.md`: over the cap is a refusal, never a truncation.
///
/// The values are those of `RenderCaps` in `python/qd_data/render.py`. They are duplicated rather
/// than shared because the two lanes are two languages, and the duplication is held together by
/// `tests/wire_context_crosslang.rs::the_escape_growth_bound_and_every_byte_cap_are_the_same_numbers_on_both_sides`,
/// which reads the real Python values at test time and compares all five.
///
/// Not by `tests/render_contract.rs`, which an earlier version of this comment credited: that file
/// pins the rendered *bytes* of one sample against a frozen golden, which is a different and also
/// useful guarantee, but it never reads a cap number. See [`ESCAPE_WORST_CASE_GROWTH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderCaps {
    pub max_context_bytes: usize,
    pub max_question_bytes: usize,
    pub max_option_bytes: usize,
    pub max_task_bytes: usize,
    pub max_rendered_bytes: usize,
}

/// A `RenderCaps` whose fields cannot express a legal request.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CapsError {
    #[error("RenderCaps.{field} must be a positive integer, got {value}")]
    NotPositive { field: &'static str, value: usize },
    #[error(
        "max_option_bytes ({option}) must not exceed max_context_bytes ({context}); keeping the \
         option cap at or below the context cap is what makes docs/hardening.md §3's \"option text \
         longer than the context cap -> refusal\" automatically true"
    )]
    OptionCapAboveContextCap { option: usize, context: usize },
    #[error(
        "max_rendered_bytes ({rendered}) is below the worst-case escaped size of a legal context \
         ({floor}); a legal request would be refused by arithmetic alone, which is its own failure"
    )]
    RenderedCapBelowFloor { rendered: usize, floor: usize },
}

impl RenderCaps {
    /// The defaults, matching `python/qd_data/render.py`'s `RenderCaps`.
    pub const DEFAULT: RenderCaps = RenderCaps {
        max_context_bytes: 131_072,
        max_question_bytes: 4_096,
        max_option_bytes: 512,
        max_task_bytes: 256,
        // Six times the context cap plus slack: see `ESCAPE_WORST_CASE_GROWTH`. Matches
        // `RenderCaps.max_rendered_bytes` in `python/qd_data/render.py`.
        max_rendered_bytes: 802_816,
    };

    /// Refuse a configuration that cannot express a legal request, rather than refusing the
    /// requests it later makes impossible.
    pub fn validate(&self) -> Result<(), CapsError> {
        for (field, value) in [
            ("max_context_bytes", self.max_context_bytes),
            ("max_question_bytes", self.max_question_bytes),
            ("max_option_bytes", self.max_option_bytes),
            ("max_task_bytes", self.max_task_bytes),
            ("max_rendered_bytes", self.max_rendered_bytes),
        ] {
            if value == 0 {
                return Err(CapsError::NotPositive { field, value });
            }
        }
        if self.max_option_bytes > self.max_context_bytes {
            return Err(CapsError::OptionCapAboveContextCap {
                option: self.max_option_bytes,
                context: self.max_context_bytes,
            });
        }
        // If the rendered cap were tighter than the worst-case escaped size, a context inside
        // `max_context_bytes` could still be refused purely by arithmetic -- a spurious
        // refusal, which is its own failure. Refuse the *configuration* instead.
        //
        // This expression is a transcription of `RenderCaps.__post_init__` in
        // `python/qd_data/render.py`, down to the 8192 of slack, so the two lanes accept and
        // reject exactly the same configurations. See `ESCAPE_WORST_CASE_GROWTH` for why the
        // factor is 6 rather than the 2 this check used until 2026-09-19.
        let floor = ESCAPE_WORST_CASE_GROWTH * self.max_context_bytes + 8192;
        if self.max_rendered_bytes < floor {
            return Err(CapsError::RenderedCapBelowFloor {
                rendered: self.max_rendered_bytes,
                floor,
            });
        }
        Ok(())
    }
}

impl Default for RenderCaps {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// -- escaping -----------------------------------------------------------------------------------

/// Characters that can produce an apparent line break, or terminate a C string, without being
/// `\n`. Transcribed from `_HEX_ESCAPED` in `python/qd_data/render.py`.
///
/// This is the **second** rule, alongside `<|`. `str.splitlines()` in Python — and several
/// tokenizer preprocessors — break on U+000B, U+000C, U+001C, U+001D, U+001E, U+0085, U+2028 and
/// U+2029 as well as on `\n` and `\r`. An option containing one of them rendered as *two* apparent
/// option lines, which is the "17th apparent option" `docs/hardening.md` §3 forbids, while a
/// `count('\n')` check said the format was intact. The same characters in the context make two
/// derivations of a line number disagree.
///
/// U+0000 is here for a different reason: it is not a line break, but the context crosses the FFI
/// boundary and a NUL truncates any C string that later carries it.
const HEX_ESCAPED: &[char] = &[
    '\u{0000}', '\u{000B}', '\u{000C}', '\u{001C}', '\u{001D}', '\u{001E}', '\u{0085}', '\u{2028}',
    '\u{2029}',
];

/// Worst-case bytes-out per byte-in for [`escape_block`] / [`escape_inline`].
///
/// The maximum is six, reached by a one-byte character that takes a six-byte `\uXXXX` escape:
/// U+0000, U+000B, U+000C, U+001C, U+001D and U+001E. Backslash, `<|`, `\n`, `\r` and `\t` are only
/// 2x, and the multi-byte members of [`HEX_ESCAPED`] are cheaper still per input byte — U+0085 is
/// two bytes in and six out (3x), U+2028 and U+2029 three bytes in and six out (2x).
///
/// This is a bound on the **bytes**, not on the characters, and [`RenderCaps::validate`] uses it so
/// that a context inside `max_context_bytes` can never be refused by arithmetic alone.
///
/// It read **2** until 2026-09-19, when 2 was the worst case and `\` -> `\\` the only multiplying
/// rule. [`HEX_ESCAPED`] made it 6 and the constant was not updated on this side, so a 131072-byte
/// context of U+000B was refused by `RenderedPromptOverCap` despite being a legal request. The
/// Python lane had already corrected its own copy to 6 and raised `max_rendered_bytes` to 802816,
/// which meant the two lanes were, for that window, refusing *different* requests. Closed as
/// `GAP-RT-ESCAPE-GROWTH-FLOOR`; `ESCAPE_WORST_CASE_GROWTH` in `python/qd_data/render.py` is the
/// other half of the pair.
///
/// # What pins the pair, and what does not
///
/// This doc comment used to end *"and `tests/render_contract.rs` pins them together"*. **It did
/// not.** That file compares frozen golden *prompt bytes* captured from one Python run on
/// 2026-09-19; it reads no Python file at test time, and its sample context contains no escapable
/// character, so it could not observe this constant even in principle. Its
/// `the_default_caps_are_self_consistent` calls only Rust's own [`RenderCaps::validate`], which
/// says nothing about the other lane. A comment claiming a guarantee that does not exist is worse
/// than no comment: it is why nobody went looking.
///
/// The real pin is
/// `tests/wire_context_crosslang.rs::the_escape_growth_bound_and_every_byte_cap_are_the_same_numbers_on_both_sides`,
/// added 2026-09-19. It runs the **real** `qd_data.render` through
/// `tests/crosslang_fixtures.py::render_caps` and compares, at test time: this constant, all five
/// [`RenderCaps::DEFAULT`] fields, and — by driving both lanes' validators at and just below
/// Python's floor — the floor *formula* rather than a second copy of the expression. Reintroducing
/// the historical 6-vs-2 split fails it, which was checked rather than assumed.
///
/// `tests/render_contract.rs` still earns its place: goldens are what catch a change in the
/// rendered *bytes*. They are simply a different guarantee from this one.
pub const ESCAPE_WORST_CASE_GROWTH: usize = 6;

/// Escape untrusted multi-line text — the context.
///
/// Guarantees: the result contains no `<|`, and has exactly as many `\n` as the input.
pub fn escape_block(text: &str) -> String {
    escape(text, false)
}

/// Escape untrusted text that must occupy exactly one line — options, the question, the task, a
/// slot name.
///
/// Guarantees: the result contains no `<|` and no newline, so it cannot become a second option
/// line.
pub fn escape_inline(text: &str) -> String {
    escape(text, true)
}

/// A single left-to-right pass. Sequential `replace` calls would be ambiguous to invert; this is
/// written as a scanner so [`unescape`] is its exact inverse.
///
/// It scans **code points**, not bytes, because [`HEX_ESCAPED`] contains U+0085, U+2028 and U+2029,
/// which are multi-byte in UTF-8. The arm order is that of `_escape` in
/// `python/qd_data/render.py`; a reordering would change the output for a character matched by two
/// arms.
fn escape(text: &str, inline: bool) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 8);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => out.push_str("\\\\"),
            // The one rule the whole defence rests on. A trailing `<` with nothing after it is an
            // ordinary character, exactly as in the Python scanner.
            '<' if chars.peek() == Some(&'|') => {
                out.push_str("<\\|");
                chars.next();
            }
            // Block mode keeps real newlines: span labels are line numbers into this text, so the
            // line count must survive escaping untouched.
            '\n' => out.push_str(if inline { "\\n" } else { "\n" }),
            // Escaped in *both* modes. A bare CR is a line break to `splitlines()` but not to
            // `split('\n')`, so leaving it raw in block mode would leave two line counts
            // disagreeing on any CR-only or mixed-ending file — the corpus `docs/hardening.md` §1
            // lists as an attack.
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str(if inline { "\\t" } else { "\t" }),
            c if HEX_ESCAPED.contains(&c) => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// An input [`unescape`] cannot invert.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnescapeError {
    #[error("dangling escape at end of input")]
    Dangling,
    #[error("unknown escape `\\{ch}` at offset {offset}")]
    Unknown { offset: usize, ch: char },
    #[error("truncated `\\u` escape at offset {offset}: `{found}` is not four hex digits")]
    BadHexEscape { offset: usize, found: String },
}

/// Exact inverse of [`escape_block`] and [`escape_inline`].
///
/// Exists so a test can assert the context region round-trips: if `unescape(region) == original`
/// for adversarial inputs, then nothing in the context was able to terminate it early or change its
/// meaning — a stronger statement than checking for any particular attack string.
///
/// An unknown escape is an error rather than being passed through: a renderer and an unescaper that
/// disagree about the alphabet is exactly the drift this module exists to prevent.
///
/// This handles `\uXXXX`, and so does `python/qd_data/render.py::unescape` — the two unescapers
/// accept exactly the same alphabet. That was `GAP-RT-PY-UNESCAPE-HEX`, opened when the Python
/// escaper had gained [`HEX_ESCAPED`] and its unescaper had not; it is closed, and
/// `python/tests/test_render.py::test_escape_round_trips_over_the_whole_break_set` is the
/// enumerated check that keeps it closed rather than leaving it to a generator to stumble on
/// U+2028.
///
/// One deliberate asymmetry remains: the Python side also refuses a `\uXXXX` escape for any code
/// point *outside* [`HEX_ESCAPED`], where this one accepts any scalar value. Both directions
/// invert their own escaper exactly; the Python one additionally refuses strings its escaper could
/// not have produced. Tightening this side would make [`UnescapeError`] carry a set membership it
/// does not need for the round-trip property, so it is named here instead of hidden.
pub fn unescape(text: &str) -> Result<String, UnescapeError> {
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    let src = text.as_bytes();
    while i < src.len() {
        if src[i] != b'\\' {
            // Copy one whole code point: `i` is always on a character boundary here, because every
            // branch below advances by a full escape sequence and this one by a full character.
            let rest = &text[i..];
            match rest.chars().next() {
                Some(c) => {
                    out.push(c);
                    i += c.len_utf8();
                }
                None => break,
            }
            continue;
        }
        if i + 1 >= src.len() {
            return Err(UnescapeError::Dangling);
        }
        match src[i + 1] {
            b'\\' => {
                out.push('\\');
                i += 2;
            }
            b'|' => {
                out.push('|');
                i += 2;
            }
            b'n' => {
                out.push('\n');
                i += 2;
            }
            b'r' => {
                out.push('\r');
                i += 2;
            }
            b't' => {
                out.push('\t');
                i += 2;
            }
            b'u' => {
                let digits = text.get(i + 2..i + 6).unwrap_or("");
                let code = u32::from_str_radix(digits, 16).ok().and_then(char::from_u32);
                match code {
                    Some(c) => {
                        out.push(c);
                        i += 6;
                    }
                    None => {
                        return Err(UnescapeError::BadHexEscape {
                            offset: i,
                            found: digits.to_string(),
                        })
                    }
                }
            }
            other => {
                return Err(UnescapeError::Unknown {
                    offset: i,
                    ch: other as char,
                })
            }
        }
    }
    Ok(out)
}

// -- rendered output ----------------------------------------------------------------------------

/// What one decode row denotes. Row order is the order the letters were rendered in, so row `i`
/// carries letter `OPTION_LETTERS[i]` — except the last, which is always [`RowLabel::Noul`] and
/// carries [`NOUL_LETTER`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowLabel {
    Choice(String),
    Score(u32),
    Noul,
}

/// One slot's decode query and the alphabet it decodes over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotRender {
    pub name: String,
    pub type_name: &'static str,
    /// The bytes appended to the prefix for this slot's 1-token query.
    pub suffix: String,
    /// Decode rows in rendered order. The last entry is always [`RowLabel::Noul`].
    ///
    /// For a `span` slot this is `[Noul]` only: a span decodes over line-start pointers, not over
    /// letters, and those rows come from the context rather than from the prompt. See
    /// [`crate::answer::span_rows`].
    pub rows: Vec<RowLabel>,
    /// For a `choice` slot, the permutation applied to the request's option order; empty when the
    /// options were rendered in the order given. The runtime's second pass sets this.
    pub permutation: Vec<usize>,
}

impl SlotRender {
    pub fn letter_for_row(&self, row: usize) -> Option<char> {
        if row + 1 == self.rows.len() {
            return Some(NOUL_LETTER as char);
        }
        OPTION_LETTERS.get(row).map(|b| *b as char)
    }
}

/// The prompt, split the way the runtime consumes it.
///
/// `docs/schema-api.md`'s answering procedure prefills the context **once**, snapshots, then
/// answers each slot as a 1-token query from the snapshot — so the prefix is rendered once and each
/// slot contributes a short suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub prefix: String,
    pub slots: Vec<SlotRender>,
    context_region: std::ops::Range<usize>,
}

impl RenderedPrompt {
    /// Byte range within [`Self::prefix`] holding the escaped context.
    ///
    /// A backend **must** encode this range with special-token handling disabled. Nothing in this
    /// crate can enforce that — it is the tokenizer's behaviour, and there is no tokenizer here —
    /// so the range is exported rather than assumed.
    pub fn context_region(&self) -> std::ops::Range<usize> {
        self.context_region.clone()
    }

    /// The escaped context exactly as it appears between the delimiters.
    pub fn context_region_str(&self) -> &str {
        &self.prefix[self.context_region.clone()]
    }

    pub fn prompt_for(&self, slot_name: &str) -> Option<String> {
        self.slots
            .iter()
            .find(|s| s.name == slot_name)
            .map(|s| format!("{}{}", self.prefix, s.suffix))
    }

    pub fn prompts(&self) -> Vec<String> {
        self.slots
            .iter()
            .map(|s| format!("{}{}", self.prefix, s.suffix))
            .collect()
    }

    pub fn slot(&self, slot_name: &str) -> Option<&SlotRender> {
        self.slots.iter().find(|s| s.name == slot_name)
    }
}

// -- the renderer -------------------------------------------------------------------------------

/// Render a validated request to the exact bytes the model sees.
///
/// The check order is that of `python/qd_data/render.py::render`, so the *first* refusal a request
/// earns is the same one in both lanes. A request that is refused here was already structurally
/// valid — [`crate::wire::validate`] ran first — so everything below is a byte-cap or an encoding
/// check, never a shape check.
pub fn render(request: &DecisionRequest, caps: &RenderCaps) -> Result<RenderedPrompt, Refusal> {
    let ctx_bytes = request.context.as_bytes();
    if ctx_bytes.len() > caps.max_context_bytes {
        return Err(Refusal::ContextOverCap {
            cap: caps.max_context_bytes,
            actual: ctx_bytes.len(),
        });
    }
    // Borrowing the bytes as `str` is a *validity check*, not a round trip: it copies nothing,
    // normalizes nothing, and re-encodes nothing. The property `Context` protects — that no
    // Unicode normalization can move a token boundary — is untouched by it.
    let ctx_text = std::str::from_utf8(ctx_bytes).map_err(|e| Refusal::ContextNotUtf8 {
        offset: e.valid_up_to(),
        invalid_len: e.error_len().unwrap_or(0),
    })?;
    if request.context.non_whitespace_len() == 0 {
        return Err(Refusal::ContextEmpty {
            len: ctx_bytes.len(),
            non_whitespace: 0,
        });
    }

    if request.question.len() > caps.max_question_bytes {
        return Err(Refusal::QuestionOverCap {
            cap: caps.max_question_bytes,
            actual: request.question.len(),
        });
    }
    if request.task.len() > caps.max_task_bytes {
        return Err(Refusal::TaskOverCap {
            cap: caps.max_task_bytes,
            actual: request.task.len(),
        });
    }

    let mut prefix = String::new();
    prefix.push_str(M_BEGIN);
    prefix.push('\n');
    prefix.push_str(M_VERSION);
    prefix.push_str(&request.schema_version.to_string());
    prefix.push('\n');
    prefix.push_str(M_TASK);
    prefix.push_str(&escape_inline(&request.task));
    prefix.push('\n');
    prefix.push_str(M_ROUTE);
    prefix.push_str(route_str(request.route));
    prefix.push('\n');
    prefix.push_str(M_QUESTION);
    prefix.push_str(&escape_inline(&request.question));
    prefix.push('\n');
    prefix.push_str(M_CTX_BEGIN);
    prefix.push('\n');
    let ctx_start = prefix.len();
    prefix.push_str(&escape_block(ctx_text));
    let ctx_end = prefix.len();
    prefix.push('\n');
    prefix.push_str(M_CTX_END);
    prefix.push('\n');

    let mut slots = Vec::with_capacity(request.slots.len());
    for (index, slot) in request.slots.iter().enumerate() {
        slots.push(slot_suffix(slot, caps, &[], index)?);
    }

    let rendered = RenderedPrompt {
        prefix,
        slots,
        context_region: ctx_start..ctx_end,
    };
    for text in rendered.prompts() {
        if text.len() > caps.max_rendered_bytes {
            return Err(Refusal::RenderedPromptOverCap {
                cap: caps.max_rendered_bytes,
                actual: text.len(),
            });
        }
    }
    Ok(rendered)
}

/// Render one `choice` slot's suffix with its options permuted.
///
/// This is the contract's **second pass**: *"run a second pass with the options permuted and
/// require agreement. Disagreement is `noul`."* The prefix is unchanged — the second pass reuses
/// the same prefill and the same snapshot, so only the suffix is re-rendered.
pub fn permuted_slot_suffix(
    slot: &SlotSpec,
    caps: &RenderCaps,
    permutation: &[usize],
    slot_index: usize,
) -> Result<SlotRender, Refusal> {
    slot_suffix(slot, caps, permutation, slot_index)
}

fn route_str(route: Route) -> &'static str {
    route.as_str()
}

fn slot_suffix(
    slot: &SlotSpec,
    caps: &RenderCaps,
    permutation: &[usize],
    slot_index: usize,
) -> Result<SlotRender, Refusal> {
    let name = slot.name().to_string();
    // `wire::parse_slot` is the only constructor of `SlotSpec` in this crate, so on the wire path
    // this has already passed. It is checked again here because the renderer also serves specs
    // built in Rust — `qd oneshot`, fixtures, tests — and a name reaching the prompt uncapped is
    // bounded only by `max_rendered_bytes`, which is a bound on the whole prompt rather than on
    // this field. One canonical owner, two call sites. `GAP-RT-SLOT-NAME-UNCAPPED`.
    check_slot_name(&name, slot_index)?;
    let type_name = slot.kind().as_str();

    let mut head = String::new();
    head.push_str(M_SLOT);
    head.push_str(&escape_inline(&name));
    head.push('\n');
    head.push_str(M_TYPE);
    head.push_str(type_name);
    head.push('\n');

    // (rendered text, row label) in rendered order, before the reserved noul row.
    let pairs: Vec<(String, RowLabel)> = match slot {
        SlotSpec::Choice { options, .. } => {
            let order: Vec<usize> = if permutation.is_empty() {
                (0..options.len()).collect()
            } else {
                permutation.to_vec()
            };
            order
                .iter()
                .filter_map(|i| options.get(*i))
                .map(|v| (v.clone(), RowLabel::Choice(v.clone())))
                .collect()
        }
        // Ordinal: letters map to bins 1..n in order, never shuffled. The bins are ordered and the
        // loss is cumulative (CORAL-style), so permuting them would destroy what the loss depends
        // on. `permutation` is ignored here by construction, not by omission.
        SlotSpec::Score { bins, .. } => (1..=*bins)
            .map(|b| (b.to_string(), RowLabel::Score(b)))
            .collect(),
        // A span is decoded by a pointer head over line-start tokens, so it has no letter
        // alphabet — but noul is present in *every* option set, so the abstain letter is still
        // rendered.
        SlotSpec::Span { .. } => Vec::new(),
    };

    let mut rows: Vec<RowLabel> = Vec::with_capacity(pairs.len() + 1);
    let mut body = String::new();
    body.push_str(M_OPT_BEGIN);
    body.push('\n');
    for (index, (text, label)) in pairs.iter().enumerate() {
        if text.len() > caps.max_option_bytes {
            return Err(Refusal::OptionTextOverCap {
                slot: name,
                option_index: index,
                cap: caps.max_option_bytes,
                actual: text.len(),
            });
        }
        let Some(letter) = OPTION_LETTERS.get(index) else {
            // Unreachable for a validated request: `wire::validate` caps options at MAX_OPTIONS and
            // bins at MAX_BINS, both 16. Refusing rather than panicking keeps the promise that no
            // production path unwraps.
            return Err(Refusal::TooManyOptions {
                slot: name,
                limit: OPTION_LETTERS.len(),
                actual: pairs.len(),
            });
        };
        body.push(*letter as char);
        body.push_str(". ");
        body.push_str(&escape_inline(text));
        body.push('\n');
        rows.push(label.clone());
    }
    body.push(NOUL_LETTER as char);
    body.push_str(". ");
    body.push_str(&escape_inline(NOUL_LABEL));
    body.push('\n');
    rows.push(RowLabel::Noul);
    body.push_str(M_OPT_END);
    body.push('\n');

    Ok(SlotRender {
        name,
        type_name,
        suffix: format!("{head}{body}{M_ANSWER}"),
        rows,
        permutation: permutation.to_vec(),
    })
}

/// Build the permutation the second pass uses for one slot.
///
/// Seeded from the request digest and the slot name, so it is reproducible for a given request: a
/// caller that sees a `noul` from permutation disagreement can replay the exact pair of passes that
/// disagreed.
///
/// # It is a derangement, and that is load-bearing
///
/// `docs/schema-api.md` says only *"a second pass with the options permuted"*. A uniform
/// permutation is **not enough**, and the gap is not theoretical: with four options, a uniform draw
/// leaves option 0 where it is a quarter of the time, and on those requests a backend answering
/// purely by letter position agrees with itself and the abstention never fires. The check would
/// then be a coin flip that looks like a guarantee.
///
/// Every option must move. This uses Sattolo's algorithm, which draws uniformly from the cyclic
/// permutations — a single `n`-cycle, so no element is fixed for `n >= 2`. Then *whatever* row wins
/// the first pass, its option sits at a different row in the second, and pure letter-position bias
/// is caught on every request rather than on three quarters of them.
///
/// `tests/answering_procedure.rs::letter_position_bias_becomes_an_abstention` is the test that
/// found this; it failed against a uniform permutation.
/// Recorded as `GAP-RT-SPEC-PERMUTATION-DERANGEMENT`.
pub fn second_pass_permutation(request_digest: &[u8; 32], slot_name: &str, n: usize) -> Vec<usize> {
    if n < 2 {
        return (0..n).collect();
    }
    let mut seed = Vec::with_capacity(32 + slot_name.len());
    seed.extend_from_slice(request_digest);
    seed.extend_from_slice(slot_name.as_bytes());
    crate::CounterRng::new("second-pass-permutation", &seed).cyclic_permutation(n)
}
