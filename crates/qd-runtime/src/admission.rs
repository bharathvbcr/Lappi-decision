//! Admission: a request whose task the release was not trained on, or whose context is not the
//! shape its task was trained on, is refused before the model is asked.
//!
//! # The task, for every request
//!
//! A request's `task` is its family id: training builds every request with `task=family_id`
//! (`python/qd_data/mixture.py::_request`). A task id outside the families the weights were
//! trained on reaches the model as a prompt shape it never saw, and gets back an answer that
//! looks like any other (GAP-RUNTIME-ADMITS-TASKS-NO-RELEASE-FAMILY-TRAINS-2026-10-03). So the
//! task is checked first, against [`TrainedFamilies`] (Fable's pipeline ruling, item 4):
//!
//! * [`TrainedFamilies::Recorded`] — the release's `trained_families`. Any other task is
//!   [`Refusal::TaskNotTrained`], naming the families there are.
//! * [`TrainedFamilies::Unrecorded`] — a release whose manifest does not record them. It cannot
//!   say what it trained, so it admits no task (`available` empty).
//! * [`TrainedFamilies::NoRelease`] — a runtime built without a release
//!   ([`crate::runtime::Runtime::with_backend`], [`crate::runtime::Runtime::build`]): there is
//!   no record to check against and the task check **does not run**. This is the test seam and
//!   the reference backend, whose every answer is `degraded`; every model path in the product is
//!   built from a release (`crates/qd-metal/src/serve.rs` through
//!   [`crate::runtime::Runtime::from_release`]). The runtime reports which of the three it
//!   holds ([`crate::runtime::Runtime::trained_families`]), so "not run" never reads as "passed"
//!   (GAP-RUNTIME-WITH-BACKEND-RUNS-NO-TRAINED-FAMILY-CHECK-2026-10-03).
//!
//! # The slot shape, for every request of a trained family
//!
//! A family trained in a handful of slot shapes (`docs/caller-contract.md` section 4), and a
//! request in another one is a prompt the model never saw read through calibration fitted on other
//! requests (GAP-QD-EXPORT-GATE-CHECKS-DECLARED-SHAPES-NOT-ADMISSION-2026-10-17). The shape is the
//! request's sorted slot keys (`choice:N`, `score:N`, `span`; N counts the reserved noul row), the
//! spelling the calibration lookup and `qd-export`'s gate use. After the task is admitted, it is
//! checked against [`TrainedShapes`]:
//!
//! * [`TrainedShapes::Recorded`] — the release's `trained_slot_shapes`, which `qd-export` writes
//!   from the train manifest's per-entry `slot_shape`. A shape not recorded for the task is
//!   [`Refusal::SlotShapeNotTrained`], naming the family, the shape sent, the shapes trained and
//!   the caller-contract section. Not part of the shape: slot names, option text and order, and
//!   the question (they are in the prompt and the label-set hash, not in the calibration key).
//! * [`TrainedShapes::NotRecorded`] — a release whose manifest has no `trained_slot_shapes` (every
//!   export before 2026-10-17). It cannot say what it trained and **the shape check does not
//!   run**: its requests are admitted on the task check alone, as before. That is "not run", not
//!   "passed", and [`crate::runtime::Runtime::trained_slot_shapes`] reports it.
//! * [`TrainedShapes::NoRelease`] — as for the task check, the test seam and the reference
//!   backend: not run.
//!
//! The shape is checked before the `code.defect_class` context: it is structural and cheap, and a
//! context's shape is a question only for a request in a shape the release can answer.
//!
//! # What is checked of the context, for which task
//!
//! Only [`DEFECT_CLASS_TASK`]. Its rows are rendered by one rewriter
//! (`python/qd_data/mixture.py::rewrite_defect_class`, `header = f"file: {raw.path}\n\n"` then
//! the diff), in two trained shapes:
//!
//! * **Single-file** (the plain corpus, and every val row): a `file: <path>` line, an empty
//!   line, then one or more unified-diff hunks. The path is the file's, and its language is
//!   checked.
//! * **Composed** (`qd-mutate compose`, the `commitpackft-composed-*` corpora v4 and v5 train
//!   on): a `file: <repo>` line, an empty line, then one or more file blocks, each opening with
//!   exactly `diff --git a/<p> b/<p>`, `--- a/<p>`, `+++ b/<p>` (the three lines
//!   `python/qd_data/defect_class.py::_composed_diff_span` requires of every block) and holding
//!   one or more hunks. `file:` names the row's repository there, not a file (`path == repo` on
//!   all 25,000 rows of `commitpackft-composed-v2`), so the language is checked per block, on
//!   `<p>`. The block count is not checked: three or more is a generation option
//!   (`min_files`), not a shape.
//!
//! The third line picks the shape: `diff --git ` opens a composed context, anything else is read
//! as single-file. The check refuses, each with a typed [`Refusal`] carrying the line, what was
//! expected and what was found:
//!
//! * no `file: ` header, a path that is not UTF-8 or is blank, or no empty line after it;
//! * a language not one the pool held ([`qd_lang::DEFECT_CLASS_POOL_LANGUAGES`], read through
//!   the pool's own [`qd_lang::language_from_path`]) — Swift included, which the pool reader
//!   maps but the built pool holds none of — on the `file:` path (single-file) or on any
//!   block's path (composed);
//! * in a single-file context, anything before the first hunk header (a bare `---`/`+++`
//!   preamble too: no trained context shows one outside a `diff --git` block), or no hunk;
//! * in a composed context, a `diff --git` line whose two paths differ or are empty, a block
//!   whose `---`/`+++` lines are not exactly `--- a/<p>` / `+++ b/<p>` (an `index`, mode or
//!   `/dev/null` line included: no trained block has one), or a block with no hunk;
//! * a hunk header that is not `@@ -a[,b] +c[,d] @@` with an optional ` section` after it;
//! * a hunk line whose first byte is not ` `, `-`, `+` or `\` (an empty line included: the
//!   trained diffs mark an empty context line with a single space);
//! * a hunk whose body has more or fewer old (` ` and `-`) or new (` ` and `+`) lines than its
//!   header declares.
//!
//! A single trailing `\n` terminates the last line and is accepted: the trained contexts drop it,
//! a caller's diff usually has it, and it opens no line.
//!
//! What this does **not** do: judge content. A diff that parses is admitted whatever is in it; a
//! scrambled diff whose hunks still parse reaches the model, which is the model's call. The OOD
//! gate suite's scoring path (`python/qd_train/ood.py`, `tools/real_ft_run.py --ood`) does not go
//! through this runtime and is unchanged.
//!
//! Admission never answers and never emits `allow`: it only refuses (CLAUDE.md rule 7).

use qd_lang::{DEFECT_CLASS_POOL_LANGUAGES, language_from_path};

use crate::context::Context;
use crate::refusal::Refusal;
use crate::release::TrainedSlotShapes;
use crate::schema::DecisionRequest;

/// Where the caller finds the shape each family is asked in; named in every
/// [`Refusal::SlotShapeNotTrained`].
pub const CONTRACT_SHAPES_SECTION: &str = "docs/caller-contract.md section 4 (Ask only trained shapes)";

/// The task whose contexts are held to the trained diff shape. The family id of
/// `python/qd_data/defect_class.py::DEFECT_FAMILY_ID`; a request's `task` is its family id
/// (`python/qd_data/mixture.py::_request`, `task=family_id`).
pub const DEFECT_CLASS_TASK: &str = "code.defect_class";

/// The first line's prefix.
const FILE_HEADER: &[u8] = b"file: ";
/// The line that opens a composed context's file block, before `a/<p> b/<p>`.
const BLOCK_HEADER: &[u8] = b"diff --git ";

/// Bytes of caller content a refusal echoes back. The context can be megabytes; a refusal that
/// carried a whole line of it would be a second copy of the payload.
const EXCERPT_BYTES: usize = 80;
/// Bytes of the path a refusal echoes back.
const PATH_EXCERPT_BYTES: usize = 256;

/// What a runtime knows about the task families its weights were trained on. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainedFamilies {
    /// The release's `trained_families`, sorted and unique: any other task is refused.
    Recorded(Vec<String>),
    /// A release whose manifest does not record what it trained: every task is refused.
    Unrecorded,
    /// No release, so nothing to check against: the task check does not run.
    NoRelease,
}

impl TrainedFamilies {
    /// What a release, or every member of an ensemble, records.
    pub fn of_release(families: Option<&[String]>) -> Self {
        match families {
            Some(families) => TrainedFamilies::Recorded(families.to_vec()),
            None => TrainedFamilies::Unrecorded,
        }
    }

    /// Whether a request's task is checked at all; `false` only for [`TrainedFamilies::NoRelease`].
    pub fn is_checked(&self) -> bool {
        !matches!(self, TrainedFamilies::NoRelease)
    }
}

/// What a runtime knows about the slot shapes each trained family trained on. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainedShapes {
    /// The release's `trained_slot_shapes`: a request of a trained family in any other shape is
    /// refused.
    Recorded(TrainedSlotShapes),
    /// A release whose manifest does not record them (every export before 2026-10-17): it cannot
    /// say what shapes it trained, and the shape check **does not run**. Reported by
    /// [`crate::runtime::Runtime::trained_slot_shapes`], so "not run" never reads as "passed".
    NotRecorded,
    /// No release: nothing to check against, and the shape check does not run.
    NoRelease,
}

impl TrainedShapes {
    /// What a release, or every member of an ensemble, records.
    pub fn of_release(shapes: Option<&TrainedSlotShapes>) -> Self {
        match shapes {
            Some(shapes) => TrainedShapes::Recorded(shapes.clone()),
            None => TrainedShapes::NotRecorded,
        }
    }

    /// The state as the `status` op reports it (`slot_shape_check`): `checked` only when a
    /// recorded set is enforced; `not_recorded` for a legacy release; `no_release` for none. The
    /// last two are the check not run and are never spelled `checked`.
    pub fn check_state(&self) -> &'static str {
        match self {
            TrainedShapes::Recorded(_) => "checked",
            TrainedShapes::NotRecorded => "not_recorded",
            TrainedShapes::NoRelease => "no_release",
        }
    }

    /// Whether a request's shape is checked at all; `true` only for [`TrainedShapes::Recorded`].
    pub fn is_checked(&self) -> bool {
        matches!(self, TrainedShapes::Recorded(_))
    }
}

/// Refuse `request` if its task is not one the weights were trained on, or if its task holds its
/// context to a trained shape and the context is not in it. The task is checked first: whether a
/// context has the shape of a task is a question only for a task the release can answer.
pub fn admit(
    request: &DecisionRequest,
    trained: &TrainedFamilies,
    shapes: &TrainedShapes,
) -> Result<(), Refusal> {
    admit_task(&request.task, trained)?;
    admit_shape(&request.task, &request.slot_shape(), shapes)?;
    if request.task == DEFECT_CLASS_TASK {
        admit_defect_context(&request.context)
    } else {
        Ok(())
    }
}

/// Refuse `task` unless `trained` records it; admit it unchecked when there is no release.
pub fn admit_task(task: &str, trained: &TrainedFamilies) -> Result<(), Refusal> {
    let available = match trained {
        TrainedFamilies::NoRelease => return Ok(()),
        TrainedFamilies::Recorded(families) if families.iter().any(|family| family == task) => {
            return Ok(());
        }
        TrainedFamilies::Recorded(families) => families.clone(),
        TrainedFamilies::Unrecorded => Vec::new(),
    };
    Err(Refusal::TaskNotTrained {
        task: task.to_string(),
        available,
    })
}

/// Refuse `shape`, the sorted slot keys of a request for `task`, unless `shapes` records it among
/// the shapes `task` trained. A recorded set with no entry for `task` refuses too: the release
/// reader makes that state impossible for a release, and a check that cannot find its own record
/// must not admit. [`TrainedShapes::NotRecorded`] and [`TrainedShapes::NoRelease`] do not check.
pub fn admit_shape(task: &str, shape: &[String], shapes: &TrainedShapes) -> Result<(), Refusal> {
    let TrainedShapes::Recorded(by_family) = shapes else {
        return Ok(());
    };
    let trained = by_family.get(task);
    if trained.is_some_and(|set| set.contains(shape)) {
        return Ok(());
    }
    Err(Refusal::SlotShapeNotTrained {
        task: task.to_string(),
        shape: shape.to_vec(),
        trained: trained.map_or_else(Vec::new, |set| set.iter().cloned().collect()),
        contract_section: CONTRACT_SHAPES_SECTION.to_string(),
    })
}

/// `bytes` as a short, escaped, lossless-where-possible excerpt for a refusal.
fn excerpt(bytes: &[u8], cap: usize) -> String {
    let (head, cut) = if bytes.len() > cap {
        (&bytes[..cap], true)
    } else {
        (bytes, false)
    };
    let text = String::from_utf8_lossy(head);
    if cut {
        format!("{text:?} (first {cap} of {} bytes)", bytes.len())
    } else {
        format!("{text:?}")
    }
}

fn not_diff(line: usize, expected: impl Into<String>, found: &[u8]) -> Refusal {
    Refusal::ContextNotUnifiedDiff {
        line,
        expected: expected.into(),
        found: excerpt(found, EXCERPT_BYTES),
    }
}

fn not_diff_end(line: usize, expected: impl Into<String>) -> Refusal {
    Refusal::ContextNotUnifiedDiff {
        line,
        expected: expected.into(),
        found: "the end of the context".to_string(),
    }
}

/// One open hunk: where its header was, what it declared, and how much of that is still owed.
struct Hunk {
    header_line: usize,
    old_declared: u64,
    new_declared: u64,
    old_left: u64,
    new_left: u64,
}

impl Hunk {
    /// Refuse a hunk whose body fell short of its header, naming the header's line.
    fn close(&self) -> Result<(), Refusal> {
        if self.old_left == 0 && self.new_left == 0 {
            return Ok(());
        }
        Err(Refusal::ContextNotUnifiedDiff {
            line: self.header_line,
            expected: format!(
                "a hunk body of {} old and {} new lines, as its header declares",
                self.old_declared, self.new_declared
            ),
            found: format!(
                "{} old and {} new lines",
                self.old_declared - self.old_left,
                self.new_declared - self.new_left
            ),
        })
    }
}

/// `digits[,digits]` at the front of `s`: the start and the count (absent = 1). Returns the rest.
fn range(s: &[u8]) -> Option<(u64, &[u8])> {
    fn number(s: &[u8]) -> Option<(u64, &[u8])> {
        let end = s
            .iter()
            .position(|b| !b.is_ascii_digit())
            .unwrap_or(s.len());
        if end == 0 {
            return None;
        }
        let digits = std::str::from_utf8(&s[..end]).ok()?;
        Some((digits.parse::<u64>().ok()?, &s[end..]))
    }
    let (_start, rest) = number(s)?;
    match rest.strip_prefix(b",") {
        Some(after) => number(after),
        None => Some((1, rest)),
    }
}

/// `@@ -a[,b] +c[,d] @@[ section]` -> `(b, d)`.
fn hunk_counts(line: &[u8]) -> Option<(u64, u64)> {
    let rest = line.strip_prefix(b"@@ -")?;
    let (old, rest) = range(rest)?;
    let rest = rest.strip_prefix(b" +")?;
    let (new, rest) = range(rest)?;
    let rest = rest.strip_prefix(b" @@")?;
    if rest.is_empty() || rest.first() == Some(&b' ') {
        Some((old, new))
    } else {
        None
    }
}

/// The trained `code.defect_class` context shape. See the module docs for every rule.
pub fn admit_defect_context(context: &Context) -> Result<(), Refusal> {
    let bytes = context.as_bytes();
    let body = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let mut lines = body.split(|b| *b == b'\n').zip(1usize..);

    // Line 1: `file: <path>`.
    let first = lines.next().map_or(&[][..], |(line, _)| line);
    let path = first
        .strip_prefix(FILE_HEADER)
        .ok_or_else(|| not_diff(1, "a `file: <path>` header line", first))?;
    let path =
        std::str::from_utf8(path).map_err(|_| not_diff(1, "a UTF-8 path after `file: `", first))?;
    if crate::is_blank(path) {
        return Err(not_diff(1, "a path after `file: `", first));
    }

    // Line 2: empty.
    match lines.next() {
        Some((b"", _)) => {}
        Some((line, n)) => {
            return Err(not_diff(n, "an empty line after the `file:` header", line));
        }
        None => return Err(not_diff_end(2, "an empty line after the `file:` header")),
    }

    // Line 3 picks the shape.
    let mut lines = lines.peekable();
    if lines
        .peek()
        .is_some_and(|(line, _)| line.starts_with(BLOCK_HEADER))
    {
        return admit_file_blocks(&mut lines);
    }
    admit_language(path)?;
    admit_hunks(&mut lines, 2, false).map(|_| ())
}

/// Refuse `path` unless its language, read through the pool's own map, is one the pool held.
fn admit_language(path: &str) -> Result<(), Refusal> {
    let language = language_from_path(path);
    if language.is_some_and(|lang| DEFECT_CLASS_POOL_LANGUAGES.contains(&lang)) {
        return Ok(());
    }
    Err(Refusal::ContextLanguageNotInPool {
        path: if path.len() > PATH_EXCERPT_BYTES {
            excerpt(path.as_bytes(), PATH_EXCERPT_BYTES)
        } else {
            path.to_string()
        },
        language: language
            .map_or("unrecognised", |lang| lang.as_str())
            .to_string(),
        pool: DEFECT_CLASS_POOL_LANGUAGES
            .iter()
            .map(|lang| lang.as_str().to_string())
            .collect(),
    })
}

/// `diff --git a/<p> b/<p>` -> `<p>`, when both paths are the same non-empty bytes. A path that
/// itself contains ` b/` is read unambiguously: the two copies are equal, so each is exactly
/// `(len - 3) / 2` bytes.
fn block_path(line: &[u8]) -> Option<&[u8]> {
    let rest = line.strip_prefix(BLOCK_HEADER)?.strip_prefix(b"a/")?;
    let n = rest.len().checked_sub(3)?;
    if n == 0 || n % 2 != 0 {
        return None;
    }
    let (path, tail) = rest.split_at(n / 2);
    (tail.strip_prefix(b" b/")? == path).then_some(path)
}

/// A composed context's file blocks, from line 3 to the end. See the module docs.
fn admit_file_blocks<'a>(
    lines: &mut impl Iterator<Item = (&'a [u8], usize)>,
) -> Result<(), Refusal> {
    const EXPECTED: &str = "a file block header `diff --git a/<path> b/<path>`";
    let mut next = lines.next();
    loop {
        let (line, n) = next.ok_or_else(|| not_diff_end(3, EXPECTED))?;
        let path = block_path(line).ok_or_else(|| not_diff(n, EXPECTED, line))?;
        for (at, prefix, side) in [
            (n + 1, &b"--- a/"[..], "old"),
            (n + 2, &b"+++ b/"[..], "new"),
        ] {
            let want = [prefix, path].concat();
            let expected = format!(
                "the block's {side}-file line `{}`",
                String::from_utf8_lossy(&want)
            );
            match lines.next() {
                Some((got, _)) if got == want.as_slice() => {}
                Some((got, m)) => return Err(not_diff(m, expected, got)),
                None => return Err(not_diff_end(at, expected)),
            }
        }
        let path =
            std::str::from_utf8(path).map_err(|_| not_diff(n, "a UTF-8 block path", line))?;
        admit_language(path)?;
        match admit_hunks(lines, n + 2, true)? {
            Some(header) => next = Some(header),
            None => return Ok(()),
        }
    }
}

/// One or more hunks, from the line after `after` until the end or, when `blocks`, until a line
/// that opens the next file block. Returns that block header (consumed) when it stopped at one.
fn admit_hunks<'a>(
    lines: &mut impl Iterator<Item = (&'a [u8], usize)>,
    after: usize,
    blocks: bool,
) -> Result<Option<(&'a [u8], usize)>, Refusal> {
    let mut open: Option<Hunk> = None;
    let mut last = after;
    let mut stopped = None;
    for (line, n) in lines.by_ref() {
        if blocks && line.starts_with(BLOCK_HEADER) {
            stopped = Some((line, n));
            break;
        }
        last = n;
        if line.starts_with(b"@@ ") {
            if let Some(hunk) = open.take() {
                hunk.close()?;
            }
            let (old, new) = hunk_counts(line)
                .ok_or_else(|| not_diff(n, "a hunk header `@@ -a[,b] +c[,d] @@`", line))?;
            open = Some(Hunk {
                header_line: n,
                old_declared: old,
                new_declared: new,
                old_left: old,
                new_left: new,
            });
            continue;
        }
        let Some(hunk) = open.as_mut() else {
            return Err(not_diff(n, "a hunk header `@@ -a[,b] +c[,d] @@`", line));
        };
        let (old, new) = match line.first() {
            Some(b' ') => (1, 1),
            Some(b'-') => (1, 0),
            Some(b'+') => (0, 1),
            // `\ No newline at end of file`: about the line before it, counted by neither side.
            Some(b'\\') => (0, 0),
            _ => {
                return Err(not_diff(
                    n,
                    "a hunk line starting with ` `, `-`, `+` or `\\`",
                    line,
                ));
            }
        };
        if old > hunk.old_left || new > hunk.new_left {
            return Err(not_diff(
                n,
                format!(
                    "a hunk header or the end: the hunk at line {} declared {} old and {} new \
                     lines and has them all",
                    hunk.header_line, hunk.old_declared, hunk.new_declared
                ),
                line,
            ));
        }
        hunk.old_left -= old;
        hunk.new_left -= new;
    }
    match (open, stopped) {
        (Some(hunk), stopped) => {
            hunk.close()?;
            Ok(stopped)
        }
        (None, Some((line, n))) => Err(not_diff(n, "a hunk header `@@ -a[,b] +c[,d] @@`", line)),
        (None, None) => Err(not_diff_end(
            last + 1,
            "a hunk header `@@ -a[,b] +c[,d] @@`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit_bytes(bytes: &[u8]) -> Result<(), Refusal> {
        admit_defect_context(&Context::from_bytes(bytes.to_vec()))
    }

    #[test]
    fn hunk_headers_parse_with_and_without_counts_and_with_a_section() {
        assert_eq!(hunk_counts(b"@@ -1,4 +1,4 @@"), Some((4, 4)));
        assert_eq!(hunk_counts(b"@@ -9 +9 @@"), Some((1, 1)));
        assert_eq!(hunk_counts(b"@@ -0,0 +1,3 @@"), Some((0, 3)));
        assert_eq!(hunk_counts(b"@@ -3,2 +3,2 @@ func Sum() {"), Some((2, 2)));
        for bad in [
            &b"@@ -1,4 +1,4"[..],
            b"@@ -1,4 +1,4 @@x",
            b"@@ -,4 +1,4 @@",
            b"@@ -1, +1,4 @@",
            b"@@ +1,4 -1,4 @@",
            b"@@ -1,4 +1,4 @@\r",
            b"@@ -99999999999999999999,1 +1,1 @@",
        ] {
            assert_eq!(hunk_counts(bad), None, "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn a_non_utf8_path_is_refused_and_a_non_utf8_body_line_is_left_to_the_renderer() {
        let refusal = admit_bytes(b"file: a\xff.rs\n\n@@ -1 +1 @@\n-x\n+y").unwrap_err();
        assert_eq!(refusal.kind(), "context_not_unified_diff");
        // The body's bytes are the renderer's to judge (`context_not_utf8`); admission reads
        // only the markers.
        admit_bytes(b"file: a.rs\n\n@@ -1 +1 @@\n-\xff\n+y").expect("markers are fine");
    }

    #[test]
    fn a_refusal_echoes_a_bounded_excerpt_not_the_payload() {
        let mut long = b"file: a.rs\n\n".to_vec();
        long.extend(std::iter::repeat_n(b'x', 10_000));
        match admit_bytes(&long).unwrap_err() {
            Refusal::ContextNotUnifiedDiff { line, found, .. } => {
                assert_eq!(line, 3);
                assert!(found.len() < 200, "{} bytes echoed", found.len());
                assert!(found.contains("of 10000 bytes"), "{found}");
            }
            other => panic!("expected context_not_unified_diff, got {other:?}"),
        }
        let path = "d/".repeat(1_000) + "x.c";
        match admit_bytes(format!("file: {path}\n\n@@ -1 +1 @@\n-x\n+y").as_bytes()).unwrap_err() {
            Refusal::ContextLanguageNotInPool { path: echoed, .. } => {
                assert!(echoed.len() < 400, "{} bytes echoed", echoed.len());
            }
            other => panic!("expected context_language_not_in_pool, got {other:?}"),
        }
    }

    #[test]
    fn a_task_is_admitted_only_by_a_record_that_names_it() {
        let recorded = TrainedFamilies::Recorded(vec![
            "code.change_scope".to_string(),
            DEFECT_CLASS_TASK.to_string(),
        ]);
        admit_task(DEFECT_CLASS_TASK, &recorded).expect("a recorded family is admitted");
        match admit_task("code.defect_clas", &recorded) {
            Err(Refusal::TaskNotTrained { task, available }) => {
                assert_eq!(task, "code.defect_clas");
                assert_eq!(available, ["code.change_scope", DEFECT_CLASS_TASK]);
            }
            other => panic!("a near miss is not a trained family: {other:?}"),
        }
        match admit_task(DEFECT_CLASS_TASK, &TrainedFamilies::Unrecorded) {
            Err(Refusal::TaskNotTrained { available, .. }) => assert!(available.is_empty()),
            other => panic!("an unrecorded release admits nothing: {other:?}"),
        }
        assert_eq!(
            TrainedFamilies::of_release(None),
            TrainedFamilies::Unrecorded
        );
        // No release: nothing to check against, and the runtime says the check is not run.
        admit_task("anything", &TrainedFamilies::NoRelease).expect("not checked");
        assert!(!TrainedFamilies::NoRelease.is_checked());
        assert!(recorded.is_checked() && TrainedFamilies::Unrecorded.is_checked());
    }

    fn shapes_of(family: &str, shapes: &[&[&str]]) -> TrainedShapes {
        let set = shapes
            .iter()
            .map(|shape| shape.iter().map(|k| k.to_string()).collect())
            .collect();
        TrainedShapes::Recorded(TrainedSlotShapes::from([(family.to_string(), set)]))
    }

    fn keys(shape: &[&str]) -> Vec<String> {
        shape.iter().map(|k| k.to_string()).collect()
    }

    #[test]
    fn a_shape_is_admitted_only_by_a_record_that_names_it_for_that_task() {
        let recorded = shapes_of("fam.a", &[&["choice:5", "span"], &["choice:3"]]);
        admit_shape("fam.a", &keys(&["choice:5", "span"]), &recorded).expect("recorded");
        admit_shape("fam.a", &keys(&["choice:3"]), &recorded).expect("recorded");
        for sent in [&["choice:5"][..], &["choice:5", "span", "span"], &["span"], &[]] {
            match admit_shape("fam.a", &keys(sent), &recorded) {
                Err(Refusal::SlotShapeNotTrained {
                    task,
                    shape,
                    trained,
                    contract_section,
                }) => {
                    assert_eq!(task, "fam.a");
                    assert_eq!(shape, keys(sent));
                    // Sorted, so the echo is stable.
                    assert_eq!(trained, [keys(&["choice:3"]), keys(&["choice:5", "span"])]);
                    assert_eq!(contract_section, CONTRACT_SHAPES_SECTION);
                }
                other => panic!("{sent:?} is not a trained shape: {other:?}"),
            }
        }
        // A record with no entry for the task cannot admit it: fail closed, naming nothing.
        match admit_shape("fam.b", &keys(&["choice:3"]), &recorded) {
            Err(Refusal::SlotShapeNotTrained { trained, .. }) => assert!(trained.is_empty()),
            other => panic!("a task with no recorded shape admits none: {other:?}"),
        }
    }

    #[test]
    fn the_shape_check_does_not_run_without_a_record_and_says_so() {
        admit_shape("fam.a", &keys(&["choice:99"]), &TrainedShapes::NotRecorded)
            .expect("not checked");
        admit_shape("fam.a", &keys(&["choice:99"]), &TrainedShapes::NoRelease).expect("not checked");
        assert!(!TrainedShapes::NotRecorded.is_checked());
        assert!(!TrainedShapes::NoRelease.is_checked());
        assert!(shapes_of("fam.a", &[&["span"]]).is_checked());
        assert_eq!(
            TrainedShapes::of_release(None),
            TrainedShapes::NotRecorded
        );
    }

    #[test]
    fn a_slot_key_is_one_a_request_can_produce() {
        use crate::release::is_slot_key;
        for good in ["span", "choice:2", "choice:17", "score:3", "score:129"] {
            assert!(is_slot_key(good), "{good}");
        }
        for bad in [
            "", "choice", "choice:", "choice:0", "choice:05", "choice:+5", "choice:5 ", "Choice:5",
            "span:1", "bins:5", "choice:-1", "choice:5:5", "score:4294967296",
        ] {
            assert!(!is_slot_key(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_empty_context_is_refused_at_line_one() {
        match admit_bytes(b"").unwrap_err() {
            Refusal::ContextNotUnifiedDiff { line, .. } => assert_eq!(line, 1),
            other => panic!("expected context_not_unified_diff, got {other:?}"),
        }
    }
}
