//! A span slot's line starts projected onto token positions: `qd_train.shards.
//! _span_token_positions`, every step of it that reads no `decode`.
//!
//! The Python stays the definition. `tools/real_tokenizer_pipeline.py::build_native_spancheck`
//! renders every span sequence a shard write will ask about, sends them here in bounded,
//! batch-shaped requests, and installs the replies as a lookup table in place of
//! `_span_token_positions` for the duration of the write. Any sequence this module does not
//! answer [`Status::Ok`] goes back to the reference, so a refusal -- its exception class and its
//! text -- is the reference's by construction. What this module must get exactly right is `Ok`:
//! the token positions, and that the reference would not have refused.
//!
//! ## What is computed, in the reference's order
//!
//! 1. `len(offsets) != len(ids)` -> [`Status::OffsetCount`].
//! 2. The largest offset end (0 when there are none) against the text's length in code points ->
//!    [`Status::Reach`].
//! 3. Each line start's token: the **lowest token index** whose span `[start, end)` contains it
//!    (`_token_indices_for_chars`). A `(0, 0)` special token or a `start >= end` span contains
//!    nothing, and the byte pieces of a split character, which all claim that character, resolve
//!    to the first. The first line start no span contains -> [`Status::LineInNoToken`].
//! 4. Two lines on one token under refuse-any -> [`Status::LinesCollapse`].
//! 5. A candidate past the last token -> [`Status::CandidatePastEnd`].
//! 6. An abstaining sequence is then `Ok`.
//! 7. The gold's first, then last, line start -> [`Status::GoldInNoToken`]; first after last ->
//!    [`Status::GoldReversed`]; a multi-line gold inside one token -> [`Status::GoldInOneToken`];
//!    past the end -> [`Status::GoldPastEnd`]; not a candidate -> [`Status::GoldNotCandidate`];
//!    under refuse-gold, a gold position another line shares -> [`Status::GoldSharesToken`].
//!
//! "The lowest token containing `c`" is computed by one sweep over the tokens in index order,
//! each claiming the code points of its span not yet claimed; a next-unclaimed pointer array
//! skips the claimed ones, so offsets that overlap pathologically stay linear rather than
//! lines x tokens.
//!
//! For an `Ok` sequence the reply also carries what the decode check
//! (`_assert_spans_decode_to_their_text`) needs that is not decoding:
//!
//! - each checked position's **run** -- the token plus the tokens after it that start inside the
//!   span claimed so far and are not empty -- as `(run_end, first, last)`, for the gold's two
//!   positions (unless it abstains) and then every candidate, the order the reference checks
//!   them in;
//! - [`REPLY_OFF_GRID`] when some line start is not a line start of the text itself, under
//!   `qd_train.artifacts.line_start_indices`: `\n` ends a line, a trailing `\n` opens none, `\r` is
//!   content, and empty text has no lines. That is `crates/qd-runtime/src/context.rs`'s
//!   `line_starts` with `line_count`'s empty case, counted in code points rather than bytes (the
//!   same grid: `\n` is one of each).
//!
//! **Decoding stays in Python.** Doing it here would need the `tokenizers` crate, a dependency
//! this crate does not have. The NFC question is Python's too: the request carries
//! `unicodedata.is_normalized("NFC", text)` and the reply echoes it as [`REPLY_NFC_UNSTABLE`],
//! because whether the reference's NFC exit fires depends on what the ids decode to.
//!
//! ## Request (`QDPSCIN1`), little-endian throughout
//!
//! ```text
//! magic 8 | n_seqs u64
//! per sequence:
//!   flags u8 (bit 0 abstains, bit 1 NFC-stable) | policy u8 (0 refuse-any, 1 refuse-gold)
//!   | reserved u16 = 0
//!   | text_bytes u32 | n_ids u32 | n_offsets u32 | n_lines u32 | span_start u32 | span_end u32
//!   | the UTF-8 text | n_offsets x (start u32, end u32) | n_lines x u32
//! ```
//!
//! Every character offset is a Python code-point index, never a byte index. An abstaining
//! sequence's span is `(0, 0)`. Only the ids' count travels: their values are read by nothing but
//! `decode`.
//!
//! ## Reply (`QDPSCOK1`)
//!
//! ```text
//! magic 8 | n_seqs u64
//! | n_seqs x (status u8 | flags u8 | reserved u16 = 0 | detail u32 | start u32 | end u32
//!            | n_candidates u32 | n_runs u32)
//! | every sequence's candidates, u32, in order
//! | every sequence's runs, (run_end u32, first u32, last u32), in order
//! ```
//!
//! A refused sequence has no candidates and no runs; `detail` names what refused it (the
//! character, token or count the reference's message names). An `Ok` sequence has one candidate
//! per line start and, unless [`REPLY_RUNS_OVER_BUDGET`], one run per checked position.

use crate::wire::{Cursor, le_u32s};

pub const INPUT_MAGIC: &[u8; 8] = b"QDPSCIN1";
pub const OUTPUT_MAGIC: &[u8; 8] = b"QDPSCOK1";
/// The Python side cuts its stream into requests of at most 256 MiB
/// (`SPANCHECK_REQUEST_BYTES`); this is four times that. The v5 corpus is ~386M tokens, ~3 GB
/// of offsets alone, so it always arrives in several requests.
pub const MAX_INPUT_BYTES: u64 = 1 << 30;
/// Sequences per request.
pub const MAX_SEQS: u64 = 1 << 24;
/// One sequence's UTF-8 text. A v5 sequence is at most ~10K tokens, ~40 KB; the per-thread
/// scratch is nine bytes a code point, so this bounds it at ~600 MB.
pub const MAX_TEXT_BYTES: u32 = 1 << 26;
/// Run-extension steps one sequence may take. An ordinary tokenization takes about one per
/// line start; past this the runs are omitted and [`REPLY_RUNS_OVER_BUDGET`] says so, which
/// sends that sequence's decode check to the reference.
pub const RUN_STEP_BUDGET: u64 = 1 << 22;

/// Request flag: the sequence abstains (`span_abstains or span_char_starts is None`).
pub const FLAG_ABSTAIN: u8 = 1;
/// Request flag: `unicodedata.is_normalized("NFC", text)`.
pub const FLAG_NFC_STABLE: u8 = 2;
pub const POLICY_REFUSE_ANY: u8 = 0;
pub const POLICY_REFUSE_GOLD: u8 = 1;

/// Reply flag: the request's abstain flag, echoed.
pub const REPLY_ABSTAIN: u8 = 1;
/// Reply flag: the request did not say NFC-stable, echoed.
pub const REPLY_NFC_UNSTABLE: u8 = 2;
/// Reply flag (`Ok` only): some line start is not a line start of the text.
pub const REPLY_OFF_GRID: u8 = 4;
/// Reply flag (`Ok` only): the runs took more than [`RUN_STEP_BUDGET`] steps and are omitted.
pub const REPLY_RUNS_OVER_BUDGET: u8 = 8;

/// Bytes of one sequence's fixed request header.
const SEQ_HEADER_BYTES: usize = 28;
/// Bytes of one sequence's fixed reply header.
pub const REPLY_SEQ_HEADER_BYTES: usize = 24;
/// "No token contains this code point" in the per-code-point owner table.
const NO_TOKEN: u32 = u32::MAX;

/// What the reference does with a sequence, as far as it can be said without decoding.
/// Each refusal names the exception class the reference raises for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Ok = 0,
    /// `ShardContractViolation`: as many offsets as ids, or not. `detail`: the offset count.
    OffsetCount = 1,
    /// `ShardContractViolation`: the offsets reach the text's end, or not. `detail`: the reach.
    Reach = 2,
    /// `UnencodableGold`: a line start in no token's span. `detail`: that code point.
    LineInNoToken = 3,
    /// `UnencodableGold`: lines share a token under refuse-any. `detail`: distinct tokens.
    LinesCollapse = 4,
    /// `ShardContractViolation`: a candidate past the last token. `detail`: the candidate.
    CandidatePastEnd = 5,
    /// `UnencodableGold`: a gold line start in no token's span. `detail`: that code point.
    GoldInNoToken = 6,
    /// `UnencodableGold`: the gold's first line maps after its last. `detail`: the first's token.
    GoldReversed = 7,
    /// `UnencodableGold`: a multi-line gold inside one token. `detail`: that token.
    GoldInOneToken = 8,
    /// `ShardContractViolation`: the gold's end past the last token. `detail`: that token.
    GoldPastEnd = 9,
    /// `UnencodableGold`: a gold position that is not a candidate. `detail`: the first such.
    GoldNotCandidate = 10,
    /// `UnencodableGold`: under refuse-gold, a gold token another line shares. `detail`: it.
    GoldSharesToken = 11,
}

/// One sequence of a parsed request, borrowing the request buffer.
#[derive(Debug)]
pub struct Seq<'b> {
    pub abstain: bool,
    pub nfc_stable: bool,
    pub refuse_gold: bool,
    pub text: &'b str,
    pub n_ids: u32,
    pub n_offsets: u32,
    /// `n_offsets x (start, end)`, little-endian u32 pairs.
    offsets: &'b [u8],
    /// `n_lines` little-endian u32 line starts.
    lines: &'b [u8],
    pub span: (u32, u32),
}

/// A parsed request.
#[derive(Debug)]
pub struct Request<'b> {
    pub seqs: Vec<Seq<'b>>,
}

/// One sequence's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub status: Status,
    pub flags: u8,
    pub detail: u32,
    pub start: u32,
    pub end: u32,
    pub candidates: Vec<u32>,
    /// `(run_end, first, last)` per checked position.
    pub runs: Vec<[u32; 3]>,
}

/// Parse and validate a request. Nothing is checked until all of it has been read.
pub fn parse(buf: &[u8]) -> Result<Request<'_>, String> {
    if buf.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "input is {} bytes; the bound is {MAX_INPUT_BYTES}",
            buf.len()
        ));
    }
    let mut c = Cursor::new(buf);
    let magic = c.take(8, "the magic")?;
    if &buf[magic] != INPUT_MAGIC {
        return Err(format!(
            "input does not start with {:?}",
            String::from_utf8_lossy(INPUT_MAGIC)
        ));
    }
    let n_seqs = c.u64("n_seqs")?;
    if n_seqs > MAX_SEQS {
        return Err(format!("{n_seqs} sequences; the bound is {MAX_SEQS}"));
    }
    let n_seqs = usize::try_from(n_seqs).map_err(|e| format!("n_seqs: {e}"))?;
    // Every sequence carries at least its fixed header, so a count the file cannot hold is
    // refused before anything is allocated for it.
    if n_seqs
        .checked_mul(SEQ_HEADER_BYTES)
        .is_none_or(|need| need > buf.len() - c.at)
    {
        return Err(format!(
            "{n_seqs} sequences cannot fit in the {} bytes that follow the header",
            buf.len() - c.at
        ));
    }
    let mut seqs = Vec::with_capacity(n_seqs);
    for s in 0..n_seqs {
        let head = c.take(4, "a sequence's flags")?;
        let (flags, policy) = (buf[head.start], buf[head.start + 1]);
        let reserved = u16::from_le_bytes([buf[head.start + 2], buf[head.start + 3]]);
        if flags & !(FLAG_ABSTAIN | FLAG_NFC_STABLE) != 0 || reserved != 0 {
            return Err(format!(
                "sequence {s}: flags {flags:#04x} and reserved {reserved}; only bits 0-1 of the \
                 flags are defined and the reserved field is 0"
            ));
        }
        let refuse_gold = match policy {
            POLICY_REFUSE_ANY => false,
            POLICY_REFUSE_GOLD => true,
            other => {
                return Err(format!(
                    "sequence {s}: policy {other} is neither refuse-any ({POLICY_REFUSE_ANY}) \
                     nor refuse-gold ({POLICY_REFUSE_GOLD})"
                ));
            }
        };
        let text_bytes = c.u32("a text length")?;
        if text_bytes > MAX_TEXT_BYTES {
            return Err(format!(
                "sequence {s}: a {text_bytes}-byte text; the bound is {MAX_TEXT_BYTES}"
            ));
        }
        let n_ids = c.u32("n_ids")?;
        let n_offsets = c.u32("n_offsets")?;
        let n_lines = c.u32("n_lines")?;
        let span = (c.u32("the span start")?, c.u32("the span end")?);
        if n_lines == 0 {
            return Err(format!(
                "sequence {s} has no line starts; a span sequence whose context has no lines is \
                 the reference's to refuse, and is never sent"
            ));
        }
        let abstain = flags & FLAG_ABSTAIN != 0;
        if abstain && span != (0, 0) {
            return Err(format!(
                "sequence {s} abstains but carries the span {span:?}; an abstaining span is (0, 0)"
            ));
        }
        let text = c.take(text_bytes as usize, "a text")?;
        let text = std::str::from_utf8(&buf[text])
            .map_err(|e| format!("sequence {s}'s text is not UTF-8: {e}"))?;
        let offsets = c.take(
            (n_offsets as usize)
                .checked_mul(8)
                .ok_or("the offset bytes overflow")?,
            "offsets",
        )?;
        let lines = c.take(
            (n_lines as usize)
                .checked_mul(4)
                .ok_or("the line-start bytes overflow")?,
            "line starts",
        )?;
        seqs.push(Seq {
            abstain,
            nfc_stable: flags & FLAG_NFC_STABLE != 0,
            refuse_gold,
            text,
            n_ids,
            n_offsets,
            offsets: &buf[offsets],
            lines: &buf[lines],
            span,
        });
    }
    if c.at != buf.len() {
        return Err(format!(
            "{} bytes follow the last of {n_seqs} sequences; refusing a file that is not \
             exactly what its header describes",
            buf.len() - c.at
        ));
    }
    Ok(Request { seqs })
}

/// Per-thread buffers, reused across that thread's sequences.
#[derive(Default)]
struct Scratch {
    offsets: Vec<(u32, u32)>,
    /// Per code point: is it `\n`.
    newline: Vec<bool>,
    /// Per code point: the lowest token whose span contains it, or [`NO_TOKEN`].
    owner: Vec<u32>,
    /// Next-unclaimed pointers over code points, with a sentinel at `n_chars`.
    next: Vec<u32>,
}

/// The first unclaimed code point at or after `x`, halving the path as it goes.
fn find(next: &mut [u32], mut x: u32) -> u32 {
    loop {
        let parent = next[x as usize];
        if parent == x {
            return x;
        }
        let grand = next[parent as usize];
        next[x as usize] = grand;
        x = grand;
    }
}

/// The lowest token containing code point `c`, if any.
fn token_of(owner: &[u32], c: u32) -> Option<u32> {
    owner.get(c as usize).copied().filter(|t| *t != NO_TOKEN)
}

/// `c` is a line start of the text, under `qd_train.artifacts.line_start_indices`.
fn on_grid(newline: &[bool], c: u32) -> bool {
    let c = c as usize;
    c < newline.len() && (c == 0 || newline[c - 1])
}

/// A count that fits a u32, or an error naming it.
fn u32_of(n: usize, what: &str) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{what} ({n}) does not fit a u32"))
}

/// The run of each of `positions`, as the decode check walks it: `(run_end, first, last)`.
/// `None` when the walk takes more than `budget` steps.
fn runs_of(
    positions: impl Iterator<Item = u32>,
    offsets: &[(u32, u32)],
    n_tokens: u32,
    budget: u64,
) -> Option<Vec<[u32; 3]>> {
    let mut steps = 0u64;
    let mut out = Vec::new();
    for pos in positions {
        let (first, mut last) = offsets[pos as usize];
        let mut run_end = pos + 1;
        while run_end < n_tokens {
            let (start, end) = offsets[run_end as usize];
            // `first <= start < last` and the token is not empty (shards.py's run loop).
            if !(first..last).contains(&start) || end <= start {
                break;
            }
            last = last.max(end);
            run_end += 1;
            steps += 1;
            if steps > budget {
                return None;
            }
        }
        out.push([run_end, first, last]);
    }
    Some(out)
}

/// One sequence's answer. `Err` only for a state the parse makes impossible.
fn check(seq: &Seq<'_>, scratch: &mut Scratch, run_budget: u64) -> Result<Outcome, String> {
    let echo = (if seq.abstain { REPLY_ABSTAIN } else { 0 })
        | (if seq.nfc_stable { 0 } else { REPLY_NFC_UNSTABLE });
    let refused = |status: Status, detail: u32| Outcome {
        status,
        flags: echo,
        detail,
        start: 0,
        end: 0,
        candidates: Vec::new(),
        runs: Vec::new(),
    };

    // 1. As many offsets as ids.
    if seq.n_offsets != seq.n_ids {
        return Ok(refused(Status::OffsetCount, seq.n_offsets));
    }
    scratch.offsets.clear();
    scratch.offsets.extend(
        seq.offsets
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                (
                    u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                    u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
                )
            }),
    );

    // 2. The offsets reach the end of the text, in code points.
    scratch.newline.clear();
    scratch.newline.extend(seq.text.chars().map(|ch| ch == '\n'));
    let n_chars = u32_of(scratch.newline.len(), "the code-point count")?;
    let reach = scratch.offsets.iter().map(|o| o.1).max().unwrap_or(0);
    if reach != n_chars {
        return Ok(refused(Status::Reach, reach));
    }

    // 3. The lowest token containing each code point. Every end is at most `n_chars` (step 2),
    //    so every claimed code point is in range.
    scratch.owner.clear();
    scratch.owner.resize(scratch.newline.len(), NO_TOKEN);
    scratch.next.clear();
    scratch.next.extend(0..=n_chars);
    for (token, &(start, end)) in (0u32..).zip(scratch.offsets.iter()) {
        if start >= end {
            continue;
        }
        let mut at = find(&mut scratch.next, start);
        while at < end {
            scratch.owner[at as usize] = token;
            scratch.next[at as usize] = at + 1;
            at = find(&mut scratch.next, at + 1);
        }
    }
    let mut candidates = Vec::with_capacity(seq.lines.len() / 4);
    for line in le_u32s(seq.lines) {
        match token_of(&scratch.owner, line) {
            Some(token) => candidates.push(token),
            None => return Ok(refused(Status::LineInNoToken, line)),
        }
    }

    // 4-5. Collapse under refuse-any, and a candidate past the end.
    let mut distinct = candidates.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let collapsed = distinct.len() != candidates.len();
    if collapsed && !seq.refuse_gold {
        return Ok(refused(
            Status::LinesCollapse,
            u32_of(distinct.len(), "the distinct-token count")?,
        ));
    }
    let highest = distinct
        .last()
        .copied()
        .ok_or("a sequence with line starts produced no candidate")?;
    if highest >= seq.n_ids {
        return Ok(refused(Status::CandidatePastEnd, highest));
    }

    let mut flags = echo;
    if !le_u32s(seq.lines).all(|line| on_grid(&scratch.newline, line)) {
        flags |= REPLY_OFF_GRID;
    }

    // 6. An abstaining sequence is answered.
    if seq.abstain {
        let runs = runs_of(
            candidates.iter().copied(),
            &scratch.offsets,
            seq.n_ids,
            run_budget,
        );
        return Ok(answered(flags, (0, 0), candidates, runs));
    }

    // 7. The gold.
    let (start_char, end_char) = seq.span;
    let Some(start) = token_of(&scratch.owner, start_char) else {
        return Ok(refused(Status::GoldInNoToken, start_char));
    };
    let Some(end) = token_of(&scratch.owner, end_char) else {
        return Ok(refused(Status::GoldInNoToken, end_char));
    };
    if start > end {
        return Ok(Outcome {
            start,
            end,
            ..refused(Status::GoldReversed, start)
        });
    }
    if start_char != end_char && start == end {
        return Ok(refused(Status::GoldInOneToken, start));
    }
    if end >= seq.n_ids {
        return Ok(refused(Status::GoldPastEnd, end));
    }
    if let Some(missing) = [start, end].into_iter().find(|p| !candidates.contains(p)) {
        return Ok(refused(Status::GoldNotCandidate, missing));
    }
    if collapsed
        && let Some(shared) = [start, end]
            .into_iter()
            .find(|p| candidates.iter().filter(|q| *q == p).count() > 1)
    {
        return Ok(refused(Status::GoldSharesToken, shared));
    }
    let runs = runs_of(
        [start, end].into_iter().chain(candidates.iter().copied()),
        &scratch.offsets,
        seq.n_ids,
        run_budget,
    );
    Ok(answered(flags, (start, end), candidates, runs))
}

/// An `Ok` outcome; runs over budget are dropped and flagged rather than sent in part.
fn answered(
    flags: u8,
    (start, end): (u32, u32),
    candidates: Vec<u32>,
    runs: Option<Vec<[u32; 3]>>,
) -> Outcome {
    let (flags, runs) = match runs {
        Some(runs) => (flags, runs),
        None => (flags | REPLY_RUNS_OVER_BUDGET, Vec::new()),
    };
    Outcome {
        status: Status::Ok,
        flags,
        detail: 0,
        start,
        end,
        candidates,
        runs,
    }
}

impl Request<'_> {
    /// Every sequence's outcome, in request order, on up to `threads` threads. The result does
    /// not depend on `threads`: each sequence is checked on its own, into its own slot.
    pub fn check_all(&self, threads: usize, run_budget: u64) -> Result<Vec<Outcome>, String> {
        if self.seqs.is_empty() {
            return Ok(Vec::new());
        }
        let threads = threads.clamp(1, self.seqs.len());
        let per = self.seqs.len().div_ceil(threads);
        let results: Vec<Result<Vec<Outcome>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .seqs
                .chunks(per)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut scratch = Scratch::default();
                        chunk
                            .iter()
                            .map(|seq| check(seq, &mut scratch, run_budget))
                            .collect::<Result<Vec<_>, String>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("a spancheck thread panicked".to_string()))
                })
                .collect()
        });
        let mut out = Vec::with_capacity(self.seqs.len());
        for chunk in results {
            out.extend(chunk?);
        }
        Ok(out)
    }
}

/// The reply's bytes.
pub fn encode_output(outcomes: &[Outcome]) -> Result<Vec<u8>, String> {
    let n_candidates: usize = outcomes.iter().map(|o| o.candidates.len()).sum();
    let n_runs: usize = outcomes.iter().map(|o| o.runs.len()).sum();
    let mut out = Vec::with_capacity(
        16 + outcomes.len() * REPLY_SEQ_HEADER_BYTES + n_candidates * 4 + n_runs * 12,
    );
    out.extend_from_slice(OUTPUT_MAGIC);
    out.extend_from_slice(
        &u64::try_from(outcomes.len())
            .map_err(|e| format!("the sequence count: {e}"))?
            .to_le_bytes(),
    );
    for o in outcomes {
        out.push(o.status as u8);
        out.push(o.flags);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&o.detail.to_le_bytes());
        out.extend_from_slice(&o.start.to_le_bytes());
        out.extend_from_slice(&o.end.to_le_bytes());
        out.extend_from_slice(&u32_of(o.candidates.len(), "a candidate count")?.to_le_bytes());
        out.extend_from_slice(&u32_of(o.runs.len(), "a run count")?.to_le_bytes());
    }
    for o in outcomes {
        for candidate in &o.candidates {
            out.extend_from_slice(&candidate.to_le_bytes());
        }
    }
    for o in outcomes {
        for run in &o.runs {
            for value in run {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// `qd-prep spancheck`: request bytes in, reply bytes and a summary line out.
pub fn run_spancheck(buf: &[u8], threads: usize) -> Result<(Vec<u8>, String), String> {
    let request = parse(buf)?;
    let outcomes = request.check_all(threads, RUN_STEP_BUDGET)?;
    let ok = outcomes.iter().filter(|o| o.status == Status::Ok).count();
    Ok((
        encode_output(&outcomes)?,
        format!(
            "qd-prep spancheck: {} sequences, {ok} ok and {} refused, on {threads} thread(s)",
            outcomes.len(),
            outcomes.len() - ok,
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sequence to put in a request.
    struct S {
        text: String,
        n_ids: Option<u32>,
        offsets: Vec<(u32, u32)>,
        lines: Vec<u32>,
        span: Option<(u32, u32)>,
        refuse_gold: bool,
        nfc_stable: bool,
    }

    impl S {
        fn new(text: &str, offsets: Vec<(u32, u32)>, lines: Vec<u32>) -> Self {
            Self {
                text: text.to_string(),
                n_ids: None,
                offsets,
                lines,
                span: None,
                refuse_gold: false,
                nfc_stable: true,
            }
        }

        fn gold(mut self, start: u32, end: u32) -> Self {
            self.span = Some((start, end));
            self
        }

        fn refuse_gold(mut self) -> Self {
            self.refuse_gold = true;
            self
        }
    }

    fn request(seqs: &[S]) -> Vec<u8> {
        let mut out = INPUT_MAGIC.to_vec();
        out.extend_from_slice(&(seqs.len() as u64).to_le_bytes());
        for s in seqs {
            let flags = (if s.span.is_none() { FLAG_ABSTAIN } else { 0 })
                | (if s.nfc_stable { FLAG_NFC_STABLE } else { 0 });
            out.push(flags);
            out.push(if s.refuse_gold {
                POLICY_REFUSE_GOLD
            } else {
                POLICY_REFUSE_ANY
            });
            out.extend_from_slice(&0u16.to_le_bytes());
            let (start, end) = s.span.unwrap_or((0, 0));
            for v in [
                s.text.len() as u32,
                s.n_ids.unwrap_or(s.offsets.len() as u32),
                s.offsets.len() as u32,
                s.lines.len() as u32,
                start,
                end,
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(s.text.as_bytes());
            for (a, b) in &s.offsets {
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
            }
            for l in &s.lines {
                out.extend_from_slice(&l.to_le_bytes());
            }
        }
        out
    }

    fn outcomes(seqs: &[S], threads: usize) -> Vec<Outcome> {
        parse(&request(seqs))
            .unwrap()
            .check_all(threads, RUN_STEP_BUDGET)
            .unwrap()
    }

    fn one(s: S) -> Outcome {
        outcomes(&[s], 1).remove(0)
    }

    /// `test_shards.byte_offsets`: every UTF-8 byte a token claiming its whole code point.
    fn byte_offsets(text: &str) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for (ci, ch) in text.chars().enumerate() {
            for _ in 0..ch.len_utf8() {
                out.push((ci as u32, ci as u32 + 1));
            }
        }
        out
    }

    /// `qd_train.artifacts.line_start_indices`, in code points.
    fn line_starts(text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0];
        for (i, ch) in chars.iter().enumerate() {
            if *ch == '\n' && i + 1 != chars.len() {
                out.push(i as u32 + 1);
            }
        }
        out
    }

    /// `test_shards._BLANK_PAIR_TEXT` and its one-token merge of "\n \n ".
    const BLANK_PAIR: &str = "ctx\n a\n \n \n+b\n c\n";

    fn blank_pair_offsets() -> Vec<(u32, u32)> {
        let chars: Vec<char> = BLANK_PAIR.chars().collect();
        let (mut out, mut i) = (Vec::new(), 0usize);
        while i < chars.len() {
            if chars[i..].starts_with(&['\n', ' ', '\n', ' ']) {
                out.push((i as u32, i as u32 + 4));
                i += 4;
            } else {
                out.push((i as u32, i as u32 + 1));
                i += 1;
            }
        }
        out
    }

    #[test]
    fn the_blank_pair_fixture_answers_as_test_shards_pins_it() {
        let lines = vec![0, 4, 7, 9, 11, 14];
        assert_eq!(line_starts(BLANK_PAIR), lines);
        let base = || S::new(BLANK_PAIR, blank_pair_offsets(), lines.clone());
        // refuse-any: the two blank lines share token 6, so the slot is refused.
        let any = one(base().gold(11, 11));
        assert_eq!((any.status, any.detail), (Status::LinesCollapse, 5));
        // refuse-gold: kept, one candidate per line, the collapsed pair sharing token 6.
        let gold = one(base().gold(11, 11).refuse_gold());
        assert_eq!(gold.status, Status::Ok);
        assert_eq!(gold.candidates, vec![0, 4, 6, 6, 8, 11]);
        assert_eq!((gold.start, gold.end), (8, 8));
        let abstain = one(base().refuse_gold());
        assert_eq!((abstain.status, abstain.flags & REPLY_ABSTAIN), (Status::Ok, REPLY_ABSTAIN));
        assert_eq!(abstain.candidates, vec![0, 4, 6, 6, 8, 11]);
        // A gold on one of the collapsed lines is refused even under refuse-gold.
        let shared = one(base().gold(9, 9).refuse_gold());
        assert_eq!((shared.status, shared.detail), (Status::GoldSharesToken, 6));
    }

    #[test]
    fn byte_pieces_resolve_to_the_first_and_their_run_covers_the_character() {
        // "x\nṢab": Ṣ is three bytes, each a token claiming (2, 3).
        let text = "x\n\u{1e62}ab";
        let offsets = byte_offsets(text);
        assert_eq!(offsets[2..5], [(2, 3), (2, 3), (2, 3)]);
        let o = one(S::new(text, offsets, vec![0, 2]).gold(2, 2));
        assert_eq!(o.status, Status::Ok);
        assert_eq!(o.candidates, vec![0, 2]);
        assert_eq!((o.start, o.end), (2, 2));
        // Checked positions: the gold twice, then the candidates; the piece run is 2..5.
        assert_eq!(o.runs, vec![[5, 2, 3], [5, 2, 3], [1, 0, 1], [5, 2, 3]]);
    }

    #[test]
    fn a_split_character_merged_into_the_previous_token_runs_like_the_reference() {
        // Qwen's shape over "is Ṣal": " Ṣ" is (2, 4), then two continuation bytes over (3, 4).
        let offsets = vec![(0, 1), (1, 2), (2, 4), (3, 4), (3, 4), (4, 5), (5, 6)];
        let o = one(S::new("is \u{1e62}al", offsets, vec![0]));
        assert_eq!(o.status, Status::Ok);
        assert_eq!(o.candidates, vec![0]);
        let wide = runs_of(
            [2u32].into_iter(),
            &[(0, 1), (1, 2), (2, 4), (3, 4), (3, 4), (4, 5), (5, 6)],
            7,
            RUN_STEP_BUDGET,
        );
        assert_eq!(wide, Some(vec![[5, 2, 4]]));
    }

    #[test]
    fn special_tokens_and_empty_spans_are_never_chosen() {
        let o = one(S::new("abcdef", vec![(0, 0), (0, 3), (0, 0), (3, 6)], vec![0, 3]));
        assert_eq!(o.candidates, vec![1, 3]);
        // A start past its end contains nothing either.
        let reversed = one(S::new("ab", vec![(2, 0), (0, 2)], vec![0]));
        assert_eq!(reversed.candidates, vec![1]);
    }

    #[test]
    fn unsorted_overlapping_offsets_take_the_lowest_index() {
        // Token 0 claims 3..5 first; token 1 covers all five code points but only 0..3 are
        // left for it; token 2 is wholly shadowed. Line 0 is token 1's, line 3 token 0's.
        let o = one(S::new("ab\ncd", vec![(3, 5), (0, 5), (0, 3)], vec![0, 3]));
        assert_eq!((o.status, o.candidates), (Status::Ok, vec![1, 0]));
    }

    #[test]
    fn every_refusal_is_the_references_first_one() {
        let text = "ab\ncd";
        let per_char = || (0..5).map(|i| (i, i + 1)).collect::<Vec<_>>();
        let mut short = S::new(text, per_char(), vec![0, 3]);
        short.n_ids = Some(4);
        let count = one(short);
        assert_eq!((count.status, count.detail), (Status::OffsetCount, 5));
        let reach = one(S::new(text, vec![(0, 2), (2, 4)], vec![0, 3]));
        assert_eq!((reach.status, reach.detail), (Status::Reach, 4));
        let none = one(S::new(text, vec![(0, 1), (2, 5)], vec![0, 1, 3]));
        assert_eq!((none.status, none.detail), (Status::LineInNoToken, 1));
        let past = one(S::new(text, per_char(), vec![0, 9]));
        assert_eq!((past.status, past.detail), (Status::LineInNoToken, 9));
        let gold_none = one(S::new(text, vec![(0, 1), (2, 5)], vec![0, 3]).gold(0, 1));
        assert_eq!((gold_none.status, gold_none.detail), (Status::GoldInNoToken, 1));
        let reversed = one(S::new(text, vec![(3, 5), (0, 3)], vec![0, 3]).gold(0, 3));
        assert_eq!(
            (reversed.status, reversed.start, reversed.end),
            (Status::GoldReversed, 1, 0)
        );
        // "a\nb\nc": token 2 holds lines 2 and 4, so a gold over both is one token.
        let one_token = one(
            S::new("a\nb\nc", vec![(0, 1), (1, 2), (2, 5)], vec![0, 2, 4])
                .gold(2, 4)
                .refuse_gold(),
        );
        assert_eq!((one_token.status, one_token.detail), (Status::GoldInOneToken, 2));
        let not_candidate = one(S::new(text, per_char(), vec![0, 3]).gold(1, 1));
        assert_eq!(
            (not_candidate.status, not_candidate.detail),
            (Status::GoldNotCandidate, 1)
        );
    }

    #[test]
    fn the_line_grid_is_pythons_in_code_points() {
        // CRLF: CR is content, so the line after it starts after the LF.
        let crlf = "a\r\nb";
        assert_eq!(line_starts(crlf), vec![0, 3]);
        let on = one(S::new(crlf, byte_offsets(crlf), vec![0, 3]));
        assert_eq!(on.flags & REPLY_OFF_GRID, 0);
        let off = one(S::new(crlf, byte_offsets(crlf), vec![0, 2]));
        assert_eq!((off.status, off.flags & REPLY_OFF_GRID), (Status::Ok, REPLY_OFF_GRID));
        // A lone CR is not a line terminator.
        let cr = "a\rb";
        assert_eq!(
            one(S::new(cr, byte_offsets(cr), vec![0, 2])).flags & REPLY_OFF_GRID,
            REPLY_OFF_GRID
        );
        // A trailing newline opens no line; an astral character before it is one code point.
        let astral = "\u{1f9ea}\nz\n";
        assert_eq!(line_starts(astral), vec![0, 2]);
        let o = one(S::new(astral, byte_offsets(astral), vec![0, 2]).gold(2, 2));
        assert_eq!((o.status, o.flags & REPLY_OFF_GRID), (Status::Ok, 0));
        assert_eq!((o.start, o.end), (5, 5));
        let trailing = one(S::new(astral, byte_offsets(astral), vec![0, 4]));
        assert_eq!(trailing.status, Status::LineInNoToken);
    }

    #[test]
    fn spans_at_the_start_and_the_end_and_an_empty_one_are_answered() {
        let text = "one\ntwo\nthree";
        let lines = line_starts(text);
        assert_eq!(lines, vec![0, 4, 8]);
        let per_char = byte_offsets(text);
        let first = one(S::new(text, per_char.clone(), lines.clone()).gold(0, 0));
        assert_eq!((first.status, first.start, first.end), (Status::Ok, 0, 0));
        let last = one(S::new(text, per_char.clone(), lines.clone()).gold(8, 8));
        assert_eq!((last.status, last.start, last.end), (Status::Ok, 8, 8));
        let whole = one(S::new(text, per_char, lines).gold(0, 8));
        assert_eq!((whole.status, whole.start, whole.end), (Status::Ok, 0, 8));
        assert_eq!(whole.runs.len(), 2 + 3);
    }

    #[test]
    fn nfc_is_echoed_never_decided() {
        let mut s = S::new("e\u{301}\nx", byte_offsets("e\u{301}\nx"), vec![0, 3]);
        s.nfc_stable = false;
        let o = one(s);
        assert_eq!((o.status, o.flags & REPLY_NFC_UNSTABLE), (Status::Ok, REPLY_NFC_UNSTABLE));
    }

    #[test]
    fn a_walk_over_budget_drops_the_runs_and_says_so() {
        // Token i covers [i, n): every run extends to the end.
        let text = "abcdefgh";
        let offsets: Vec<(u32, u32)> = (0..8).map(|i| (i, 8)).collect();
        let lines = vec![0];
        let seq = request(&[S::new(text, offsets.clone(), lines)]);
        let req = parse(&seq).unwrap();
        let within = req.check_all(1, 100).unwrap().remove(0);
        assert_eq!(within.runs, vec![[8, 0, 8]]);
        let over = req.check_all(1, 3).unwrap().remove(0);
        assert_eq!(over.status, Status::Ok);
        assert_eq!(over.flags & REPLY_RUNS_OVER_BUDGET, REPLY_RUNS_OVER_BUDGET);
        assert!(over.runs.is_empty());
        assert_eq!(over.candidates, within.candidates);
    }

    #[test]
    fn answers_do_not_depend_on_the_thread_count() {
        let texts: Vec<String> = (0..23)
            .map(|i| format!("row {i}\n\u{e9}\u{1f9ea} {i}\n\n  tail {i}\n"))
            .collect();
        let seqs = || {
            texts
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let s = S::new(t, byte_offsets(t), line_starts(t));
                    if i % 3 == 0 {
                        s
                    } else {
                        let lines = line_starts(t);
                        s.gold(lines[0], lines[lines.len() - 1])
                    }
                })
                .collect::<Vec<_>>()
        };
        let base = outcomes(&seqs(), 1);
        assert!(base.iter().all(|o| o.status == Status::Ok));
        for threads in [2, 5, 64] {
            assert_eq!(outcomes(&seqs(), threads), base, "{threads} threads");
        }
    }

    #[test]
    fn malformed_requests_are_refused() {
        let good = request(&[S::new("ab", vec![(0, 1), (1, 2)], vec![0])]);
        assert!(parse(&good).is_ok());
        let mut magic = good.clone();
        magic[0] = b'X';
        assert!(parse(&magic).unwrap_err().contains("does not start"));
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(parse(&trailing).unwrap_err().contains("follow the last"));
        for cut in [7, 15, 20, good.len() - 1] {
            assert!(parse(&good[..cut]).is_err(), "truncated at {cut}");
        }
        let at = 16; // the first sequence's flags byte
        let mut flags = good.clone();
        flags[at] |= 0x80;
        assert!(parse(&flags).unwrap_err().contains("only bits 0-1"));
        let mut policy = good.clone();
        policy[at + 1] = 2;
        assert!(parse(&policy).unwrap_err().contains("policy 2"));
        let mut many = good.clone();
        many[8..16].copy_from_slice(&(MAX_SEQS + 1).to_le_bytes());
        assert!(parse(&many).unwrap_err().contains("the bound is"));
        let mut more = good.clone();
        more[8..16].copy_from_slice(&1000u64.to_le_bytes());
        assert!(parse(&more).unwrap_err().contains("cannot fit"));
        let mut not_utf8 = good.clone();
        let text_at = 16 + SEQ_HEADER_BYTES;
        not_utf8[text_at] = 0xFF;
        assert!(parse(&not_utf8).unwrap_err().contains("not UTF-8"));
        let no_lines = request(&[S::new("ab", vec![(0, 2)], vec![])]);
        assert!(parse(&no_lines).unwrap_err().contains("no line starts"));
        let mut abstain_span = good.clone();
        abstain_span[at + 20..at + 24].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse(&abstain_span).unwrap_err().contains("abstains but"));
        let mut huge_text = good.clone();
        huge_text[at + 4..at + 8].copy_from_slice(&(MAX_TEXT_BYTES + 1).to_le_bytes());
        assert!(parse(&huge_text).unwrap_err().contains("the bound is"));
    }

    #[test]
    fn the_reply_is_headers_then_candidates_then_runs() {
        let o = one(S::new("ab\ncd", byte_offsets("ab\ncd"), vec![0, 3]).gold(3, 3));
        let bytes = encode_output(std::slice::from_ref(&o)).unwrap();
        assert_eq!(&bytes[..8], OUTPUT_MAGIC);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1);
        let head = &bytes[16..16 + REPLY_SEQ_HEADER_BYTES];
        assert_eq!((head[0], head[2], head[3]), (Status::Ok as u8, 0, 0));
        let word = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().unwrap());
        assert_eq!((word(8), word(12), word(16), word(20)), (3, 3, 2, 4));
        assert_eq!(
            bytes.len(),
            16 + REPLY_SEQ_HEADER_BYTES + 2 * 4 + 4 * 12,
            "two candidates, four runs"
        );
        let refused = one(S::new("ab", vec![(0, 1)], vec![0]));
        let bytes = encode_output(&[refused]).unwrap();
        assert_eq!(bytes.len(), 16 + REPLY_SEQ_HEADER_BYTES);
        assert_eq!(bytes[16], Status::Reach as u8);
    }
}
