//! Derivation B of the span label: a textual before/after line diff, with the changed range read
//! off the hunk.
//!
//! This module deliberately knows **nothing** about the mutator. Its only inputs are two strings.
//! It shares the *convention* with [`crate::span`] — 1-based, inclusive both ends, over the
//! post-mutation file — and shares no code path with it. That is the whole point: agreement
//! between the two is evidence, and a shared helper would make it a tautology.

use crate::span::LineSpan;

/// Split for diffing, under the same convention [`crate::span::line_of`] counts by.
///
/// A text ending in `\n` yields a trailing empty element — the phantom final line. Dropping it here
/// and not there is precisely the off-by-one this crate exists to prevent, so it is kept.
fn lines_for_diff(text: &str) -> Vec<&str> {
    text.split('\n').collect()
}

/// **Derivation B.** The changed line range, in post-mutation coordinates.
///
/// Returns `None` when the two texts are identical — a mutation that changed nothing is not a
/// mutation, and an example claiming a span over an empty diff is a poisoned label.
///
/// The hunk is minimised from both ends (longest common prefix, then longest common suffix), which
/// is the same range a unified diff would print. A pure deletion leaves an empty after-range; its
/// span is the single line containing the join, matching derivation A's rule for an empty byte
/// range.
pub fn span_from_text_diff(before: &str, after: &str) -> Option<LineSpan> {
    if before == after {
        return None;
    }
    let b = lines_for_diff(before);
    let a = lines_for_diff(after);

    let mut prefix = 0usize;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }

    let max_suffix = b.len().min(a.len()) - prefix;
    let mut suffix = 0usize;
    while suffix < max_suffix
        && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let changed_start = prefix; // 0-based, inclusive
    let changed_end = a.len() - suffix; // 0-based, exclusive

    if changed_end <= changed_start {
        // Pure deletion: nothing in the after-text belongs to the hunk. The span is the line the
        // deletion joined, clamped into the file. `a.len()` is at least 1 for any string, so the
        // clamp can only ever pull the line back to a real one.
        let line = u32::try_from(changed_start.min(a.len().saturating_sub(1)) + 1).ok()?;
        return Some(LineSpan::new(line, line));
    }

    let start = u32::try_from(changed_start + 1).ok()?;
    let end = u32::try_from(changed_end).ok()?;
    Some(LineSpan::new(start, end.max(start)))
}

/// A unified-style diff of `before` against `after` with **exactly one hunk**: the longest common
/// prefix and suffix are trimmed and everything between them is the hunk, however much of it is
/// unchanged.
///
/// This is the v2 renderer, kept byte-for-byte so `qd-mutate generate --single-hunk` reproduces
/// `commitpackft-corpus-v2`. It is not what a reader of real diffs sees: two edits forty lines
/// apart come out as one hunk carrying the forty unchanged lines as context, so every diff it
/// renders has one `@@` header and a task that asks "which hunk is the defect in" has nothing to
/// choose between. [`unified_multi`] is the renderer that splits.
///
/// Line numbers in the header are 1-based in each file's own coordinates.
pub fn unified(before: &str, after: &str, context: usize) -> String {
    let b = lines_for_diff(before);
    let a = lines_for_diff(after);
    let mut prefix = 0usize;
    while prefix < b.len() && prefix < a.len() && b[prefix] == a[prefix] {
        prefix += 1;
    }
    let max_suffix = b.len().min(a.len()) - prefix;
    let mut suffix = 0usize;
    while suffix < max_suffix && b[b.len() - 1 - suffix] == a[a.len() - 1 - suffix] {
        suffix += 1;
    }
    let b_start = prefix.saturating_sub(context);
    let b_end = (b.len() - suffix + context).min(b.len());
    let a_start = prefix.saturating_sub(context);
    let a_end = (a.len() - suffix + context).min(a.len());

    let mut out = String::new();
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        b_start + 1,
        b_end.saturating_sub(b_start),
        a_start + 1,
        a_end.saturating_sub(a_start)
    ));
    for line in b.get(b_start..prefix).unwrap_or_default() {
        out.push_str(&format!(" {line}\n"));
    }
    for line in b.get(prefix..b.len() - suffix).unwrap_or_default() {
        out.push_str(&format!("-{line}\n"));
    }
    for line in a.get(prefix..a.len() - suffix).unwrap_or_default() {
        out.push_str(&format!("+{line}\n"));
    }
    for line in a.get(a.len() - suffix..a_end).unwrap_or_default() {
        out.push_str(&format!(" {line}\n"));
    }
    out
}

// --- The multi-hunk renderer -----------------------------------------------------------------

/// Lines one side of a [`unified_multi`] diff may hold. Checked before anything is allocated per
/// line; over it the diff is refused, never truncated — a truncated diff is a diff of a different
/// file. The pool's largest file is 163 lines and the parser refuses anything over 10 MB, so this
/// is a ceiling on a malformed input, not a limit any real record approaches.
pub const MAX_DIFF_LINES: usize = 50_000;

/// Largest edit distance (lines deleted plus lines inserted) [`unified_multi`] will compute.
///
/// Myers is O((N+M)·D) in time; this caps D, so the worst case is bounded by
/// `2 · MAX_DIFF_LINES · MAX_EDIT_DISTANCE` snake steps rather than by the input. The refusal is
/// exact — a diff is refused if and only if its minimal edit distance exceeds this — and it is a
/// function of the two texts alone, never of a clock, so it cannot make a run irreproducible.
pub const MAX_EDIT_DISTANCE: usize = 8_192;

/// Why [`unified_multi`] declined to render a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffRefusal {
    /// One side has more than [`MAX_DIFF_LINES`] lines.
    TooManyLines {
        side: &'static str,
        lines: usize,
        cap: usize,
    },
    /// The minimal edit script is longer than [`MAX_EDIT_DISTANCE`].
    EditDistanceOverCap { cap: usize },
    /// The middle-snake search returned a split that would not shrink the problem. Not reachable
    /// for inputs whose ends were trimmed first; refused rather than looped on if it ever is.
    NoProgress,
}

impl std::fmt::Display for DiffRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffRefusal::TooManyLines { side, lines, cap } => {
                write!(f, "the {side} text has {lines} lines, over the {cap}-line cap")
            }
            DiffRefusal::EditDistanceOverCap { cap } => {
                write!(f, "the minimal edit script is longer than the {cap}-line cap")
            }
            DiffRefusal::NoProgress => {
                write!(f, "the middle-snake search split the problem without shrinking it")
            }
        }
    }
}

impl std::error::Error for DiffRefusal {}

/// One run of a line-level edit script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    Equal,
    Delete,
    Insert,
}

/// `len` lines of `op`, starting at 0-based line `old` of the before-text and `new` of the
/// after-text. A `Delete` consumes before-lines only and an `Insert` after-lines only, so the other
/// coordinate is where the run sits on that side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditRun {
    pub op: EditOp,
    pub old: usize,
    pub new: usize,
    pub len: usize,
}

/// A maximal stretch of changed lines: before-lines `old_start..old_end` were replaced by
/// after-lines `new_start..new_end` (0-based, half-open). One side may be empty — a pure insertion
/// or a pure deletion — and then its range is the position the change sits at on that side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangeBlock {
    pub old_start: usize,
    pub old_end: usize,
    pub new_start: usize,
    pub new_end: usize,
}

/// One `@@` hunk: the line ranges it covers on each side (0-based, half-open, context included)
/// and the change blocks it carries, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HunkRange {
    pub old_start: usize,
    pub old_end: usize,
    pub new_start: usize,
    pub new_end: usize,
    pub blocks: Vec<ChangeBlock>,
}

/// A rendered multi-hunk diff, with the structure it was rendered from so a caller can ask where a
/// span sits without parsing the text back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiHunkDiff {
    pub text: String,
    pub hunks: Vec<HunkRange>,
    /// Lines in the after-text, phantom final line included — at least one for any string.
    pub new_lines: usize,
}

impl MultiHunkDiff {
    /// The hunk whose after-side range contains every line of `span` (1-based, inclusive, over the
    /// after-text), if one does. `None` when the span reaches outside every hunk or straddles two —
    /// in either case some line of it is not, or not only, inside one hunk of the text a model
    /// reads, so "the hunk the defect is in" has no single answer.
    pub fn hunk_containing(&self, span: LineSpan) -> Option<usize> {
        let (first, last) = span_rows(span)?;
        self.hunks
            .iter()
            .position(|h| h.new_start <= first && last < h.new_end)
    }

    /// Whether the diff shows a change **at** `span`: an inserted line inside it, or a deletion
    /// positioned at one of its lines. A span over lines the diff carries only as context is a
    /// label over text the reader is told did not change — which is what a mutation that undoes the
    /// commit's own edit at that site produces once the diff is taken from the pre-image.
    pub fn shows_change_at(&self, span: LineSpan) -> bool {
        let Some((first, last)) = span_rows(span) else {
            return false;
        };
        self.hunks.iter().flat_map(|h| h.blocks.iter()).any(|b| {
            if b.new_end > b.new_start {
                b.new_start <= last && first < b.new_end
            } else {
                // A pure deletion sits between after-lines `new_start - 1` and `new_start`; it is
                // at the span when the line it precedes is one of the span's. That is exactly how
                // the span of a pure-deletion mutation is placed: on the line the deletion joined.
                // A deletion past the last line — the end of a file with no trailing newline —
                // joined onto that last line, and `span_from_text_diff` clamps it there too.
                let joined = b.new_start.min(self.new_lines.saturating_sub(1));
                first <= joined && joined <= last
            }
        })
    }
}

/// `span` as 0-based inclusive rows, or `None` for a malformed one.
fn span_rows(span: LineSpan) -> Option<(usize, usize)> {
    let first = usize::try_from(span.start).ok()?.checked_sub(1)?;
    let last = usize::try_from(span.end).ok()?.checked_sub(1)?;
    (first <= last).then_some((first, last))
}

/// A unified diff of `before` against `after` that splits into one hunk per cluster of changes.
///
/// Line-level and minimal: the edit script is Myers' (an O((N+M)·D) shortest edit script, here in
/// its linear-space middle-snake form), so the hunks are the ones `diff -u` would print up to
/// tie-breaking between equally short scripts. Changes separated by more than `2 · context`
/// unchanged lines are separate hunks; closer ones share a hunk, exactly as `diff -U<context>`
/// merges them. Headers are always `@@ -a,b +c,d @@`, the form `qd_train.mutate_adapter`'s
/// `_HUNK_HEADER` reads.
///
/// Lines are split the way [`span_from_text_diff`] and [`crate::span::line_of`] count them, phantom
/// final line included, so the header's new-side numbers are the span label's line numbers.
/// Within one change block every `-` line precedes every `+` line. Identical inputs give the empty
/// string.
pub fn unified_multi(before: &str, after: &str, context: usize) -> Result<String, DiffRefusal> {
    unified_multi_detailed(before, after, context).map(|d| d.text)
}

/// [`unified_multi`], returning the hunk structure beside the text.
pub fn unified_multi_detailed(
    before: &str,
    after: &str,
    context: usize,
) -> Result<MultiHunkDiff, DiffRefusal> {
    let old = lines_for_diff(before);
    let new = lines_for_diff(after);
    for (side, lines) in [("before", old.len()), ("after", new.len())] {
        if lines > MAX_DIFF_LINES {
            return Err(DiffRefusal::TooManyLines {
                side,
                lines,
                cap: MAX_DIFF_LINES,
            });
        }
    }
    let (old_ids, new_ids) = intern(&old, &new);
    let script = edit_script(&old_ids, &new_ids, MAX_EDIT_DISTANCE)?;
    let blocks = change_blocks(&script);
    let hunks = group_hunks(&blocks, context, old.len(), new.len());
    let text = render(&old, &new, &hunks);
    Ok(MultiHunkDiff {
        text,
        hunks,
        new_lines: new.len(),
    })
}

/// Map every distinct line to a small integer, so the diff compares integers. Only equality is
/// ever asked of the ids, so the order they are handed out in cannot reach the output.
fn intern<'a>(old: &[&'a str], new: &[&'a str]) -> (Vec<u32>, Vec<u32>) {
    let mut ids: std::collections::HashMap<&'a str, u32> = std::collections::HashMap::new();
    let mut id_of = |line: &'a str| -> u32 {
        // At most 2 * MAX_DIFF_LINES distinct lines reach here, far inside u32.
        let next = ids.len() as u32;
        *ids.entry(line).or_insert(next)
    };
    let old_ids = old.iter().map(|l| id_of(l)).collect();
    let new_ids = new.iter().map(|l| id_of(l)).collect();
    (old_ids, new_ids)
}

/// Appends runs, merging a run into the previous one when it continues it.
struct ScriptBuilder {
    runs: Vec<EditRun>,
}

impl ScriptBuilder {
    fn push(&mut self, run: EditRun) {
        if run.len == 0 {
            return;
        }
        if let Some(last) = self.runs.last_mut()
            && last.op == run.op
        {
            let continues = match run.op {
                EditOp::Equal => last.old + last.len == run.old && last.new + last.len == run.new,
                EditOp::Delete => last.old + last.len == run.old,
                EditOp::Insert => last.new + last.len == run.new,
            };
            if continues {
                last.len += run.len;
                return;
            }
        }
        self.runs.push(run);
    }
}

/// The outcome of one middle-snake search.
enum Bisect {
    /// The optimal path passes through `(x, y)`; diff the two quadrants separately.
    Split(usize, usize),
    /// No line in common: the whole range is a deletion followed by an insertion.
    NoCommon,
    /// The edit distance of this sub-problem exceeds the cap.
    OverCap,
}

/// Work still to do, in a stack, so recursion depth is never the call stack's problem.
enum Task {
    Solve {
        old: (usize, usize),
        new: (usize, usize),
    },
    Emit(EditRun),
}

/// A minimal line-level edit script from `old` to `new`, refused when its length exceeds `max_d`.
pub(crate) fn edit_script(old: &[u32], new: &[u32], max_d: usize) -> Result<Vec<EditRun>, DiffRefusal> {
    // A search depth that finds every script of length <= max_d at the top level: the middle
    // snake of a D-path is met by iteration ceil(D/2), and the loop runs iterations 0..limit.
    let half_cap = max_d / 2 + 2;
    let mut forward: Vec<isize> = Vec::new();
    let mut reverse: Vec<isize> = Vec::new();
    let mut out = ScriptBuilder { runs: Vec::new() };
    let mut stack = vec![Task::Solve {
        old: (0, old.len()),
        new: (0, new.len()),
    }];
    while let Some(task) = stack.pop() {
        let (old_range, new_range) = match task {
            Task::Emit(run) => {
                out.push(run);
                continue;
            }
            Task::Solve { old, new } => (old, new),
        };
        let (mut o0, mut o1) = old_range;
        let (mut n0, mut n1) = new_range;

        let prefix_old = o0;
        let prefix_new = n0;
        while o0 < o1 && n0 < n1 && old[o0] == new[n0] {
            o0 += 1;
            n0 += 1;
        }
        let prefix = o0 - prefix_old;
        let mut suffix = 0usize;
        while o0 < o1 && n0 < n1 && old[o1 - 1] == new[n1 - 1] {
            o1 -= 1;
            n1 -= 1;
            suffix += 1;
        }

        // Pushed in reverse of the order they must come out in.
        stack.push(Task::Emit(EditRun {
            op: EditOp::Equal,
            old: o1,
            new: n1,
            len: suffix,
        }));
        if o0 == o1 {
            stack.push(Task::Emit(EditRun {
                op: EditOp::Insert,
                old: o0,
                new: n0,
                len: n1 - n0,
            }));
        } else if n0 == n1 {
            stack.push(Task::Emit(EditRun {
                op: EditOp::Delete,
                old: o0,
                new: n0,
                len: o1 - o0,
            }));
        } else {
            match bisect(&old[o0..o1], &new[n0..n1], half_cap, &mut forward, &mut reverse) {
                Bisect::Split(x, y) => {
                    if (x == 0 && y == 0) || (x == o1 - o0 && y == n1 - n0) {
                        return Err(DiffRefusal::NoProgress);
                    }
                    stack.push(Task::Solve {
                        old: (o0 + x, o1),
                        new: (n0 + y, n1),
                    });
                    stack.push(Task::Solve {
                        old: (o0, o0 + x),
                        new: (n0, n0 + y),
                    });
                }
                Bisect::NoCommon => {
                    stack.push(Task::Emit(EditRun {
                        op: EditOp::Insert,
                        old: o1,
                        new: n0,
                        len: n1 - n0,
                    }));
                    stack.push(Task::Emit(EditRun {
                        op: EditOp::Delete,
                        old: o0,
                        new: n0,
                        len: o1 - o0,
                    }));
                }
                Bisect::OverCap => return Err(DiffRefusal::EditDistanceOverCap { cap: max_d }),
            }
        }
        stack.push(Task::Emit(EditRun {
            op: EditOp::Equal,
            old: prefix_old,
            new: prefix_new,
            len: prefix,
        }));
    }
    let script = out.runs;
    let distance: usize = script
        .iter()
        .filter(|r| r.op != EditOp::Equal)
        .map(|r| r.len)
        .sum();
    if distance > max_d {
        return Err(DiffRefusal::EditDistanceOverCap { cap: max_d });
    }
    Ok(script)
}

/// Myers' middle snake over `a` and `b`, both non-empty with differing first and last elements.
///
/// The forward and reverse searches advance one edit at a time until their furthest-reaching paths
/// overlap; the overlap point lies on an optimal path, so the two quadrants it separates can be
/// solved independently. Space is linear in `min(len, half_cap)`. `half_cap` bounds the number of
/// iterations, which is what bounds the time.
fn bisect(
    a: &[u32],
    b: &[u32],
    half_cap: usize,
    forward: &mut Vec<isize>,
    reverse: &mut Vec<isize>,
) -> Bisect {
    let n = a.len() as isize;
    let m = b.len() as isize;
    let max_d = (n + m + 1) / 2;
    let limit = max_d.min(half_cap as isize);
    let offset = limit + 1;
    let width = (2 * limit + 3) as usize;
    forward.clear();
    forward.resize(width, -1);
    reverse.clear();
    reverse.resize(width, -1);
    forward[(offset + 1) as usize] = 0;
    reverse[(offset + 1) as usize] = 0;
    let delta = n - m;
    // With an odd delta the forward path is the one that can complete the overlap.
    let front = delta % 2 != 0;
    let (mut k1_start, mut k1_end, mut k2_start, mut k2_end) = (0isize, 0isize, 0isize, 0isize);

    for d in 0..limit {
        let mut k1 = -d + k1_start;
        while k1 <= d - k1_end {
            let k1_at = (offset + k1) as usize;
            let mut x1 = if k1 == -d || (k1 != d && forward[k1_at - 1] < forward[k1_at + 1]) {
                forward[k1_at + 1]
            } else {
                forward[k1_at - 1] + 1
            };
            let mut y1 = x1 - k1;
            while x1 < n && y1 < m && a[x1 as usize] == b[y1 as usize] {
                x1 += 1;
                y1 += 1;
            }
            forward[k1_at] = x1;
            if x1 > n {
                // Ran off the right edge of the edit graph.
                k1_end += 2;
            } else if y1 > m {
                // Ran off the bottom edge.
                k1_start += 2;
            } else if front {
                let k2_at = offset + delta - k1;
                if k2_at >= 0 && (k2_at as usize) < width && reverse[k2_at as usize] != -1 {
                    let x2 = n - reverse[k2_at as usize];
                    if x1 >= x2 {
                        return Bisect::Split(x1 as usize, y1 as usize);
                    }
                }
            }
            k1 += 2;
        }

        let mut k2 = -d + k2_start;
        while k2 <= d - k2_end {
            let k2_at = (offset + k2) as usize;
            let mut x2 = if k2 == -d || (k2 != d && reverse[k2_at - 1] < reverse[k2_at + 1]) {
                reverse[k2_at + 1]
            } else {
                reverse[k2_at - 1] + 1
            };
            let mut y2 = x2 - k2;
            while x2 < n && y2 < m && a[(n - x2 - 1) as usize] == b[(m - y2 - 1) as usize] {
                x2 += 1;
                y2 += 1;
            }
            reverse[k2_at] = x2;
            if x2 > n {
                // Ran off the left edge.
                k2_end += 2;
            } else if y2 > m {
                // Ran off the top edge.
                k2_start += 2;
            } else if !front {
                let k1_at = offset + delta - k2;
                if k1_at >= 0 && (k1_at as usize) < width && forward[k1_at as usize] != -1 {
                    let x1 = forward[k1_at as usize];
                    let y1 = offset + x1 - k1_at;
                    if x1 >= n - x2 {
                        return Bisect::Split(x1 as usize, y1 as usize);
                    }
                }
            }
            k2 += 2;
        }
    }
    if limit < max_d {
        Bisect::OverCap
    } else {
        Bisect::NoCommon
    }
}

/// The maximal stretches of non-`Equal` runs.
fn change_blocks(script: &[EditRun]) -> Vec<ChangeBlock> {
    let mut blocks: Vec<ChangeBlock> = Vec::new();
    let mut open: Option<ChangeBlock> = None;
    for run in script {
        match run.op {
            EditOp::Equal => {
                if let Some(block) = open.take() {
                    blocks.push(block);
                }
            }
            EditOp::Delete | EditOp::Insert => {
                let block = open.get_or_insert(ChangeBlock {
                    old_start: run.old,
                    old_end: run.old,
                    new_start: run.new,
                    new_end: run.new,
                });
                match run.op {
                    EditOp::Delete => block.old_end = run.old + run.len,
                    _ => block.new_end = run.new + run.len,
                }
            }
        }
    }
    if let Some(block) = open {
        blocks.push(block);
    }
    blocks
}

/// Group change blocks into hunks: a gap of at most `2 · context` unchanged lines keeps two blocks
/// in one hunk, because their context windows would touch or overlap.
fn group_hunks(
    blocks: &[ChangeBlock],
    context: usize,
    old_len: usize,
    new_len: usize,
) -> Vec<HunkRange> {
    let mut hunks: Vec<HunkRange> = Vec::new();
    for block in blocks {
        if let Some(hunk) = hunks.last_mut()
            && let Some(prev) = hunk.blocks.last()
            && block.old_start - prev.old_end <= 2 * context
        {
            hunk.blocks.push(*block);
            hunk.old_end = (block.old_end + context).min(old_len);
            hunk.new_end = (block.new_end + context).min(new_len);
            continue;
        }
        hunks.push(HunkRange {
            old_start: block.old_start.saturating_sub(context),
            old_end: (block.old_end + context).min(old_len),
            new_start: block.new_start.saturating_sub(context),
            new_end: (block.new_end + context).min(new_len),
            blocks: vec![*block],
        });
    }
    hunks
}

/// A header's `start,len` for one side. A side with no lines in the hunk names the line before
/// it, as `diff -u` does; with any context at all that never arises, because every hunk then
/// carries at least one unchanged line on each side.
fn header_range(start: usize, end: usize) -> String {
    let len = end - start;
    if len == 0 {
        format!("{start},0")
    } else {
        format!("{},{len}", start + 1)
    }
}

fn render(old: &[&str], new: &[&str], hunks: &[HunkRange]) -> String {
    let mut out = String::new();
    for hunk in hunks {
        out.push_str(&format!(
            "@@ -{} +{} @@\n",
            header_range(hunk.old_start, hunk.old_end),
            header_range(hunk.new_start, hunk.new_end)
        ));
        let mut at = hunk.old_start;
        for block in &hunk.blocks {
            for line in &old[at..block.old_start] {
                out.push_str(&format!(" {line}\n"));
            }
            for line in &old[block.old_start..block.old_end] {
                out.push_str(&format!("-{line}\n"));
            }
            for line in &new[block.new_start..block.new_end] {
                out.push_str(&format!("+{line}\n"));
            }
            at = block.old_end;
        }
        for line in &old[at..hunk.old_end] {
            out.push_str(&format!(" {line}\n"));
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod oracle {
    /// Apply a unified diff to `before`, checking every context and `-` line against it and every
    /// header's counts and new-side start against what was consumed. The round-trip oracle: it
    /// shares nothing with the renderer but the format. Test-only, and shared with `generate`'s
    /// tests so a generated example's diff is checked by the same oracle.
    pub(crate) fn apply(before: &str, diff: &str) -> Result<String, String> {
        let old: Vec<&str> = before.split('\n').collect();
        let mut out: Vec<&str> = Vec::new();
        let mut at = 0usize;
        let mut lines = diff.split('\n').peekable();
        let mut hunk: Option<(usize, usize, usize, usize)> = None; // old/new: expected, seen
        let close = |h: Option<(usize, usize, usize, usize)>| -> Result<(), String> {
            match h {
                Some((eo, so, en, sn)) if eo != so || en != sn => Err(format!(
                    "header promised -{eo} +{en} lines, hunk carried -{so} +{sn}"
                )),
                _ => Ok(()),
            }
        };
        while let Some(line) = lines.next() {
            if line.is_empty() {
                if lines.peek().is_some() {
                    return Err("blank line inside the diff".to_string());
                }
                break;
            }
            let (lead, rest) = line.split_at(1);
            match lead {
                "@" => {
                    close(hunk.take())?;
                    let inner = line
                        .strip_prefix("@@ -")
                        .and_then(|s| s.strip_suffix(" @@"))
                        .ok_or_else(|| format!("bad header {line:?}"))?;
                    let (o, n) = inner.split_once(" +").ok_or("bad header")?;
                    let parse = |s: &str| -> Result<(usize, usize), String> {
                        let (a, b) = s.split_once(',').ok_or("header without a count")?;
                        Ok((
                            a.parse().map_err(|_| "bad start")?,
                            b.parse().map_err(|_| "bad len")?,
                        ))
                    };
                    let (os, ol) = parse(o)?;
                    let (ns, nl) = parse(n)?;
                    let old_index = if ol == 0 { os } else { os.checked_sub(1).ok_or("old start 0")? };
                    if old_index < at {
                        return Err(format!("hunk at old line {os} overlaps the previous one"));
                    }
                    out.extend_from_slice(old.get(at..old_index).ok_or("hunk past the end")?);
                    at = old_index;
                    let expect_new = if nl == 0 { out.len() } else { out.len() + 1 };
                    if ns != expect_new {
                        return Err(format!("new-side start {ns}, expected {expect_new}"));
                    }
                    hunk = Some((ol, 0, nl, 0));
                }
                " " | "-" => {
                    let h = hunk.as_mut().ok_or("content before any header")?;
                    if old.get(at) != Some(&rest) {
                        return Err(format!("line {:?} does not match before line {}", rest, at + 1));
                    }
                    at += 1;
                    h.1 += 1;
                    if lead == " " {
                        out.push(rest);
                        h.3 += 1;
                    }
                }
                "+" => {
                    let h = hunk.as_mut().ok_or("content before any header")?;
                    out.push(rest);
                    h.3 += 1;
                }
                _ => return Err(format!("unknown lead {lead:?}")),
            }
        }
        close(hunk)?;
        out.extend_from_slice(&old[at..]);
        Ok(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::oracle::apply;

    #[test]
    fn identical_texts_have_no_span() {
        assert_eq!(span_from_text_diff("a\nb\n", "a\nb\n"), None);
    }

    #[test]
    fn a_single_changed_line_is_that_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\n", "a\nX\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn a_pure_deletion_is_the_join_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\n", "a\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn an_insertion_is_the_inserted_line() {
        assert_eq!(
            span_from_text_diff("a\nc\n", "a\nb\nc\n"),
            Some(LineSpan::new(2, 2))
        );
    }

    #[test]
    fn a_change_on_the_last_line_without_a_trailing_newline() {
        assert_eq!(span_from_text_diff("a\nb", "a\nX"), Some(LineSpan::new(2, 2)));
    }

    #[test]
    fn deleting_the_whole_file_clamps_to_line_one() {
        assert_eq!(span_from_text_diff("a\nb\n", ""), Some(LineSpan::new(1, 1)));
    }

    #[test]
    fn a_multi_line_replacement_covers_every_changed_line() {
        assert_eq!(
            span_from_text_diff("a\nb\nc\nd\n", "a\nX\nY\nd\n"),
            Some(LineSpan::new(2, 3))
        );
    }

    // --- unified_multi ------------------------------------------------------------------------

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// `n` numbered lines, each distinct, ending in a newline.
    fn numbered(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("line {i}")).collect()
    }

    fn join(lines: &[String]) -> String {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    }

    fn headers(diff: &str) -> Vec<&str> {
        diff.lines().filter(|l| l.starts_with("@@")).collect()
    }

    #[test]
    fn two_edits_far_apart_are_two_hunks_and_the_single_hunk_renderer_makes_one() {
        let before = numbered(30);
        let mut after = before.clone();
        after[4] = "CHANGED 5".to_string();
        after[24] = "CHANGED 25".to_string();
        let (b, a) = (join(&before), join(&after));

        let multi = unified_multi(&b, &a, 3).expect("renders");
        assert_eq!(
            multi,
            "@@ -2,7 +2,7 @@\n line 2\n line 3\n line 4\n-line 5\n+CHANGED 5\n line 6\n line 7\n \
             line 8\n@@ -22,7 +22,7 @@\n line 22\n line 23\n line 24\n-line 25\n+CHANGED 25\n \
             line 26\n line 27\n line 28\n"
        );
        assert_eq!(apply(&b, &multi).as_deref(), Ok(a.as_str()));
        // The v2 renderer, for contrast: one header over everything between the two edits.
        assert_eq!(headers(&unified(&b, &a, 3)).len(), 1);
    }

    #[test]
    fn changes_six_lines_apart_share_a_hunk_and_seven_apart_do_not() {
        // `diff -U3` merges when the gap is at most 2 * context.
        for (gap, expected) in [(6usize, 1usize), (7, 2)] {
            let before = numbered(40);
            let mut after = before.clone();
            after[10] = "X".to_string();
            after[10 + gap + 1] = "Y".to_string();
            let (b, a) = (join(&before), join(&after));
            let diff = unified_multi(&b, &a, 3).expect("renders");
            assert_eq!(headers(&diff).len(), expected, "gap {gap}:\n{diff}");
            assert_eq!(apply(&b, &diff).as_deref(), Ok(a.as_str()));
        }
    }

    #[test]
    fn identical_texts_render_nothing() {
        assert_eq!(unified_multi("a\nb\n", "a\nb\n", 3).as_deref(), Ok(""));
        assert_eq!(unified_multi("", "", 3).as_deref(), Ok(""));
    }

    #[test]
    fn a_single_contiguous_edit_renders_as_the_single_hunk_renderer_does() {
        // One cluster of changes is one hunk either way; the two renderers differ only when there
        // is something between two clusters to split on.
        let before = join(&numbered(20));
        let after = before.replace("line 9\n", "line nine\nline 9.5\n");
        assert_eq!(
            unified_multi(&before, &after, 3).expect("renders"),
            unified(&before, &after, 3)
        );
    }

    #[test]
    fn a_hunk_at_the_start_and_one_at_the_end_clamp_their_context() {
        let before = join(&numbered(20));
        let after = before
            .replacen("line 1\n", "first\n", 1)
            .replace("line 20\n", "last\n");
        let diff = unified_multi(&before, &after, 3).expect("renders");
        // The phantom final line after the trailing newline is ordinary context.
        assert_eq!(headers(&diff), ["@@ -1,4 +1,4 @@", "@@ -17,5 +17,5 @@"]);
        assert_eq!(apply(&before, &diff).as_deref(), Ok(after.as_str()));
    }

    #[test]
    fn a_change_block_lists_its_removals_before_its_additions() {
        let diff = unified_multi("a\nb\nc\nd\n", "a\nX\nY\nd\n", 1).expect("renders");
        assert_eq!(diff, "@@ -1,4 +1,4 @@\n a\n-b\n-c\n+X\n+Y\n d\n");
    }

    #[test]
    fn context_zero_names_the_line_before_an_empty_side() {
        let diff = unified_multi("a\nb\n", "a\nX\nb\n", 0).expect("renders");
        assert_eq!(diff, "@@ -1,0 +2,1 @@\n+X\n");
        assert_eq!(apply("a\nb\n", &diff).as_deref(), Ok("a\nX\nb\n"));
    }

    #[test]
    fn too_many_lines_is_refused_not_truncated() {
        let big = "x\n".repeat(MAX_DIFF_LINES);
        assert_eq!(
            unified_multi(&big, "x\n", 3),
            Err(DiffRefusal::TooManyLines {
                side: "before",
                lines: MAX_DIFF_LINES + 1,
                cap: MAX_DIFF_LINES
            })
        );
        assert!(matches!(
            unified_multi("x\n", &big, 3),
            Err(DiffRefusal::TooManyLines { side: "after", .. })
        ));
    }

    #[test]
    fn the_edit_distance_cap_is_exact() {
        // Ten distinct lines against ten other distinct lines: D = 20 exactly.
        let old: Vec<u32> = (0..10).collect();
        let new: Vec<u32> = (100..110).collect();
        assert!(edit_script(&old, &new, 20).is_ok());
        assert_eq!(
            edit_script(&old, &new, 19),
            Err(DiffRefusal::EditDistanceOverCap { cap: 19 })
        );
        // And with lines in common, where the middle snake (not the no-common path) answers.
        let old: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let new: Vec<u32> = vec![1, 20, 3, 40, 5, 60, 7, 80, 9];
        assert!(edit_script(&old, &new, 8).is_ok());
        assert_eq!(
            edit_script(&old, &new, 7),
            Err(DiffRefusal::EditDistanceOverCap { cap: 7 })
        );
    }

    #[test]
    fn a_large_rewrite_is_refused_by_the_cap_and_a_large_file_with_a_small_edit_is_not() {
        let before = join(&numbered(MAX_EDIT_DISTANCE));
        let rewritten: String = before.replace("line", "row");
        assert_eq!(
            unified_multi(&before, &rewritten, 3),
            Err(DiffRefusal::EditDistanceOverCap {
                cap: MAX_EDIT_DISTANCE
            })
        );
        let mut after = numbered(MAX_EDIT_DISTANCE);
        after[100] = "edit".to_string();
        let after = join(&after);
        let diff = unified_multi(&before, &after, 3).expect("a one-line edit is cheap");
        assert_eq!(apply(&before, &diff).as_deref(), Ok(after.as_str()));
    }

    #[test]
    fn hunk_containing_and_shows_change_at_answer_over_after_lines() {
        let before = numbered(30);
        let mut after = before.clone();
        after[4] = "CHANGED 5".to_string();
        after.remove(20); // deletes "line 21"; after line 21 is now "line 22"
        let d = unified_multi_detailed(&join(&before), &join(&after), 3).expect("renders");
        assert_eq!(d.hunks.len(), 2);
        assert_eq!(d.hunk_containing(LineSpan::new(5, 5)), Some(0));
        assert_eq!(d.hunk_containing(LineSpan::new(21, 21)), Some(1));
        assert_eq!(d.hunk_containing(LineSpan::new(5, 21)), None, "straddles two hunks");
        assert_eq!(d.hunk_containing(LineSpan::new(12, 12)), None, "in no hunk");
        assert!(d.shows_change_at(LineSpan::new(5, 5)));
        assert!(d.shows_change_at(LineSpan::new(21, 21)), "the join line of a deletion");
        assert!(!d.shows_change_at(LineSpan::new(4, 4)), "context is not a change");
        assert!(!d.shows_change_at(LineSpan::new(22, 22)), "the line after the join is not");
        assert!(!d.shows_change_at(LineSpan::new(0, 3)), "a malformed span shows nothing");
    }

    #[test]
    fn a_deletion_that_ends_a_file_with_no_trailing_newline_shows_at_its_last_line() {
        // Deleting "\nX" from "a\nb\nX" leaves "a\nb": the deleted line sits past the last line
        // of `after`, and `span_from_text_diff` clamps the span onto that last line ("b"). The
        // diff must say a change is shown there, or every such candidate is refused as an
        // invisible needle.
        let (before, after) = ("a\nb\nX", "a\nb");
        let span = span_from_text_diff(before, after).expect("a change");
        assert_eq!(span, LineSpan::new(2, 2));
        let d = unified_multi_detailed(before, after, 3).expect("renders");
        assert!(d.shows_change_at(span), "{d:?}");
        assert_eq!(d.hunk_containing(span), Some(0));
        assert!(!d.shows_change_at(LineSpan::new(1, 1)), "the clamp moves onto the last line only");
    }

    /// Length of the longest common subsequence, by the textbook O(NM) table.
    fn lcs(a: &[u32], b: &[u32]) -> usize {
        let mut row = vec![0usize; b.len() + 1];
        for x in a {
            let mut diag = 0usize;
            for (j, y) in b.iter().enumerate() {
                let up = row[j + 1];
                row[j + 1] = if x == y { diag + 1 } else { up.max(row[j]) };
                diag = up;
            }
        }
        row[b.len()]
    }

    fn random_text(rng: &mut ChaCha20Rng, alphabet: &[&str], max_lines: usize) -> Vec<String> {
        let n = rng.random_range(0..=max_lines);
        (0..n)
            .map(|_| alphabet[rng.random_range(0..alphabet.len())].to_string())
            .collect()
    }

    /// Edit `base` in a few random places, so the pair is similar the way a commit's two images
    /// are, rather than two unrelated strings.
    fn perturb(rng: &mut ChaCha20Rng, base: &[String], alphabet: &[&str]) -> Vec<String> {
        let mut out = base.to_vec();
        for _ in 0..rng.random_range(0..6) {
            let at = rng.random_range(0..=out.len());
            match rng.random_range(0..3) {
                0 => out.insert(at, alphabet[rng.random_range(0..alphabet.len())].to_string()),
                1 if at < out.len() => {
                    out.remove(at);
                }
                _ if at < out.len() => {
                    out[at] = alphabet[rng.random_range(0..alphabet.len())].to_string();
                }
                _ => {}
            }
        }
        out
    }

    #[test]
    fn round_trip_minimality_and_shape_over_generated_inputs() {
        // Small alphabets force many equal lines and many equally short scripts, which is where a
        // middle-snake implementation goes wrong. The empty string is in every alphabet because a
        // blank source line must still render as a one-space context line.
        let alphabets: [&[&str]; 3] = [
            &["a", "b", ""],
            &["fn f() {", "}", "    x += 1;", "", "    return x;", "// c"],
            &["p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", ""],
        ];
        let mut rng = ChaCha20Rng::seed_from_u64(0x0d1f_f5ee);
        let mut multi_hunk_cases = 0usize;
        for case in 0..4000 {
            let alphabet = alphabets[case % alphabets.len()];
            let base = random_text(&mut rng, alphabet, 40);
            let other = if rng.random_bool(0.7) {
                perturb(&mut rng, &base, alphabet)
            } else {
                random_text(&mut rng, alphabet, 40)
            };
            let mut before = base.join("\n");
            let mut after = other.join("\n");
            if rng.random_bool(0.8) {
                before.push('\n');
            }
            if rng.random_bool(0.8) {
                after.push('\n');
            }
            let context = rng.random_range(0..=4);
            let d = unified_multi_detailed(&before, &after, context)
                .unwrap_or_else(|e| panic!("case {case}: refused: {e}"));

            // Applying the diff to `before` yields `after`, byte for byte.
            assert_eq!(
                apply(&before, &d.text).as_deref(),
                Ok(after.as_str()),
                "case {case} (context {context}):\nbefore {before:?}\nafter {after:?}\n{}",
                d.text
            );

            // The script is minimal: D = N + M - 2 * LCS.
            let old = lines_for_diff(&before);
            let new = lines_for_diff(&after);
            let (old_ids, new_ids) = intern(&old, &new);
            let script = edit_script(&old_ids, &new_ids, MAX_EDIT_DISTANCE).expect("small");
            let distance: usize = script
                .iter()
                .filter(|r| r.op != EditOp::Equal)
                .map(|r| r.len)
                .sum();
            assert_eq!(
                distance,
                old.len() + new.len() - 2 * lcs(&old_ids, &new_ids),
                "case {case}: not a shortest edit script"
            );

            // Shape: empty exactly when equal; otherwise newline-terminated, no blank line, every
            // header in the form the Python walker reads, hunks ordered and split only on gaps
            // wider than 2 * context.
            assert_eq!(d.text.is_empty(), before == after, "case {case}");
            if !d.text.is_empty() {
                assert!(d.text.ends_with('\n'), "case {case}");
                assert!(!d.text.contains("\n\n"), "case {case}: blank line");
            }
            assert_eq!(headers(&d.text).len(), d.hunks.len(), "case {case}");
            for h in headers(&d.text) {
                assert!(
                    h.starts_with("@@ -") && h.ends_with(" @@") && h.matches(',').count() == 2,
                    "case {case}: header {h:?}"
                );
            }
            for pair in d.hunks.windows(2) {
                let gap_old = pair[1].blocks[0].old_start - pair[0].blocks.last().expect("b").old_end;
                assert!(gap_old > 2 * context, "case {case}: hunks {pair:?} should have merged");
                assert!(pair[0].old_end <= pair[1].old_start, "case {case}: hunks overlap");
            }
            for h in &d.hunks {
                for pair in h.blocks.windows(2) {
                    assert!(pair[1].old_start - pair[0].old_end <= 2 * context, "case {case}");
                }
            }
            if d.hunks.len() >= 2 {
                multi_hunk_cases += 1;
            }
        }
        assert!(
            multi_hunk_cases > 400,
            "only {multi_hunk_cases} generated cases split, so splitting was barely exercised"
        );
    }
}
