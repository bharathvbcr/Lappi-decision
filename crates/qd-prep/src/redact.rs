//! The redactor for real job postings (v2 plan section 6, gate 2).
//!
//! A posting carries a recruiter's name, email and phone number. None may enter a training row.
//! [`redact`] replaces emails, phone numbers and signature-block names with placeholders;
//! [`residual`] reads the **redacted** text with a stricter eye and names what is still there,
//! and the caller refuses the row if it names anything. A row is also refused when redaction
//! leaves it with too few words to be a posting ([`MIN_WORDS_AFTER`]), because a posting reduced
//! to placeholders carries nothing to learn from and its label would be unsupported.
//!
//! * **Emails.** `local@domain.tld` with the at sign as `@`, a fullwidth `＠`, `﹫`, or an
//!   obfuscation: `[at]`, `(at)`, `{at}`, `<at>` in any case and with spaces inside the brackets,
//!   or the word ` at ` when a dot follows in the domain; the dots of the domain may be `.`,
//!   `[dot]`, `(dot)`, `{dot}`, `<dot>` or the word ` dot `. The local part is Unicode
//!   alphanumerics and ``._%+-'``.
//! * **Phone numbers.** A run of digits (ASCII, fullwidth, Arabic-Indic, Devanagari) joined by
//!   spaces, `-`, `.`, `/`, parentheses and dashes, with an optional leading `+`, that is a phone
//!   number by the rules at [`phone_spans`]: international (`+44 20 7946 0958`), national
//!   (`(415) 555-1234`, `07911 123456`) and unseparated (`4155551234`). Years, ISO dates,
//!   `100000-150000` pay ranges and versions are kept.
//! * **Names.** The name that follows a sign-off line (`Best regards,` / `Thanks,` ...), a name
//!   on the same line as a sign-off, and a name after a contact label (`Recruiter:`, `Posted by`,
//!   `Hiring manager -`). A name is one to four capitalised alphabetic tokens (Unicode, with `-`,
//!   `'` and initials).
//! * **Secrets** are not redacted; [`crate::vani_pool`] refuses a row that has one, through the
//!   repository's one secret screen (`qd_runtime::caller_record::secret_like_prefix`).
//!
//! Over-redaction is the accepted error: a salary range that looks like a phone number is
//! removed, a posting that mentions `work at example.com` loses it. Under-redaction is not: any
//! `@` left in the text, and any span these detectors still find after redaction, refuses the row.

use crate::pyunicode;

pub const EMAIL_PLACEHOLDER: &str = "[EMAIL]";
pub const PHONE_PLACEHOLDER: &str = "[PHONE]";
pub const NAME_PLACEHOLDER: &str = "[NAME]";

/// A redacted posting keeps at least this many words that are not placeholders.
pub const MIN_WORDS_AFTER: usize = 8;
/// Longest run a phone candidate may be (characters); longer is not scanned as one number.
const MAX_PHONE_RUN: usize = 40;
const MAX_NAME_TOKENS: usize = 4;

const AT_SIGNS: [char; 3] = ['@', '\u{ff20}', '\u{fe6b}'];

/// What [`redact`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Redacted {
    pub text: String,
    pub emails: usize,
    pub phones: usize,
    pub names: usize,
}

fn is_local(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-' | '\'')
}

fn is_label(c: char) -> bool {
    c.is_alphanumeric() || c == '-'
}

fn eq_ci(c: char, want: char) -> bool {
    c.to_lowercase().next() == Some(want)
}

fn word_at(chars: &[char], i: usize, word: &str) -> Option<usize> {
    let mut j = i;
    for w in word.chars() {
        if j < chars.len() && eq_ci(chars[j], w) {
            j += 1;
        } else {
            return None;
        }
    }
    Some(j)
}

fn skip_spaces(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
        i += 1;
    }
    i
}

/// A bracketed word `[at]` `(at)` `{at}` `<at>` (any case, spaces allowed inside) at `i`.
fn bracketed(chars: &[char], i: usize, word: &str) -> Option<usize> {
    let close = match chars.get(i)? {
        '[' => ']',
        '(' => ')',
        '{' => '}',
        '<' => '>',
        _ => return None,
    };
    let j = word_at(chars, skip_spaces(chars, i + 1), word)?;
    let j = skip_spaces(chars, j);
    (chars.get(j) == Some(&close)).then_some(j + 1)
}

/// The end of an at-token starting at `i`: `@`, a fullwidth form, a bracketed `at`, or the word
/// ` at ` (the caller requires a dot in the domain for the word form). Spaces around a symbolic
/// or bracketed token belong to it.
fn at_token(chars: &[char], i: usize) -> Option<usize> {
    let j = skip_spaces(chars, i);
    if let Some(c) = chars.get(j) {
        if AT_SIGNS.contains(c) {
            return Some(skip_spaces(chars, j + 1));
        }
        if let Some(end) = bracketed(chars, j, "at") {
            return Some(skip_spaces(chars, end));
        }
    }
    // The word form needs a space before and after: " at ".
    if j > i
        && let Some(end) = word_at(chars, j, "at")
        && chars.get(end) == Some(&' ')
    {
        return Some(skip_spaces(chars, end));
    }
    None
}

/// The end of a dot-token at `i`: `.`, a bracketed `dot`, or the word ` dot `.
fn dot_token(chars: &[char], i: usize) -> Option<usize> {
    if chars.get(i) == Some(&'.') {
        return Some(i + 1);
    }
    let j = skip_spaces(chars, i);
    if let Some(end) = bracketed(chars, j, "dot") {
        return Some(skip_spaces(chars, end));
    }
    if j > i
        && let Some(end) = word_at(chars, j, "dot")
        && chars.get(end) == Some(&' ')
    {
        return Some(skip_spaces(chars, end));
    }
    None
}

/// The email spans of `chars` as `(start, end)`, in order and disjoint.
fn email_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut floor = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let Some(after_at) = at_token(chars, i) else {
            i += 1;
            continue;
        };
        let mut start = i;
        while start > floor && is_local(chars[start - 1]) {
            start -= 1;
        }
        // A space-led token starts at the space; the local part ends just before it.
        let local = &chars[start..i];
        if local.is_empty() || !local.iter().any(|c| c.is_alphanumeric()) {
            i += 1;
            continue;
        }
        let mut j = after_at;
        let mut labels = 0usize;
        let mut last_len = 0usize;
        loop {
            let begin = j;
            while j < chars.len() && is_label(chars[j]) {
                j += 1;
            }
            if j == begin {
                break;
            }
            labels += 1;
            last_len = j - begin;
            match dot_token(chars, j) {
                Some(end) if end < chars.len() && is_label(chars[end]) => j = end,
                _ => break,
            }
        }
        // Two labels is a dotted domain. That is what keeps the word form (` at `) from reading
        // "looking at remote roles" as an email.
        if labels >= 2 && last_len >= 2 {
            spans.push((start, j));
            floor = j;
            i = j;
        } else {
            i += 1;
        }
    }
    spans
}

/// The value of a decimal digit in the scripts postings use, or `None`.
fn digit(c: char) -> Option<u32> {
    let cp = u32::from(c);
    for zero in [0x30u32, 0xff10, 0x660, 0x6f0, 0x966] {
        if (zero..zero + 10).contains(&cp) {
            return Some(cp - zero);
        }
    }
    None
}

fn is_phone_sep(c: char) -> bool {
    matches!(c, ' ' | '-' | '.' | '/' | '(' | ')' | '\u{2013}' | '\u{2014}' | '\u{2011}' | '\u{a0}')
}

fn is_dash(c: char) -> bool {
    matches!(c, '-' | '\u{2013}' | '\u{2014}' | '\u{2011}')
}

const PHONE_KEYWORDS: [&str; 10] = [
    "tel", "telephone", "phone", "call", "cell", "mobile", "fax", "whatsapp", "text", "contact",
];

/// A keyword as a whole word in the 20 characters before `start` ("context" is not "text").
fn keyword_before(chars: &[char], start: usize) -> bool {
    let from = start.saturating_sub(20);
    let before: String = chars[from..start].iter().collect::<String>().to_lowercase();
    before
        .split(|c: char| !c.is_alphabetic())
        .any(|w| PHONE_KEYWORDS.contains(&w))
}

/// The digit groups of a candidate run: maximal digit sequences.
fn groups(run: &[char]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut n = 0usize;
    for c in run {
        if digit(*c).is_some() {
            n += 1;
        } else if n > 0 {
            out.push(n);
            n = 0;
        }
    }
    if n > 0 {
        out.push(n);
    }
    out
}

/// Whether the candidate run `run` (starting at `start` in `chars`) is a phone number.
///
/// Phone: a keyword (`tel`, `phone`, `call`, `cell`, `mobile`, `fax`, `whatsapp`, `text`,
/// `contact`) in the 20 characters before it with at least seven digits; or a leading `+` or `(`
/// with seven or more digits; or ten or more digits that are not a hyphenated pair of groups (a
/// pay range, `100000-150000`); or seven to nine digits with at least two separators that are
/// not a year range, ISO or US date, or a dotted version.
fn is_phone(chars: &[char], start: usize, run: &[char]) -> bool {
    let g = groups(run);
    let d: usize = g.iter().sum();
    if d < 7 {
        return false;
    }
    if keyword_before(chars, start) {
        return true;
    }
    let seps = run.iter().filter(|c| digit(**c).is_none() && !matches!(c, '+')).count();
    let leading = run.first().copied();
    if matches!(leading, Some('+' | '(')) {
        return true;
    }
    let only_dashes = run.iter().filter(|c| digit(**c).is_none()).all(|c| is_dash(*c));
    let hyphen_pair = g.len() == 2 && seps == 1 && only_dashes;
    let date_shape = matches!(g.as_slice(), [4, 2, 2] | [2, 2, 4])
        && run.iter().filter(|c| digit(**c).is_none()).all(|c| matches!(c, '-' | '/' | '.'));
    let year_range = g.iter().all(|n| *n == 4) && only_dashes;
    let version = run.iter().filter(|c| digit(**c).is_none()).all(|c| *c == '.')
        && g.iter().all(|n| *n <= 2);
    if date_shape || year_range || version {
        return false;
    }
    if d >= 10 {
        return !hyphen_pair;
    }
    seps >= 2
}

/// The phone-number spans of `chars`. Candidates start at a digit, `+` or `(`, are not preceded
/// by an alphanumeric (`abc123456789` is an identifier), end on a digit or `)`, and are not
/// followed by one.
fn phone_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let starts = digit(c).is_some()
            || (c == '+' && chars.get(i + 1).is_some_and(|n| digit(*n).is_some() || *n == ' ' || *n == '('))
            || (c == '(' && chars.get(i + 1).is_some_and(|n| digit(*n).is_some()));
        let bounded = i == 0 || !chars[i - 1].is_alphanumeric();
        if !starts || !bounded {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && j - i < MAX_PHONE_RUN && (digit(chars[j]).is_some() || is_phone_sep(chars[j])) {
            j += 1;
        }
        // Trim trailing separators: a number ends on a digit or a closing parenthesis.
        let mut end = j;
        while end > i + 1 && !(digit(chars[end - 1]).is_some() || chars[end - 1] == ')') {
            end -= 1;
        }
        let tail_ok = end >= chars.len() || !chars[end].is_alphanumeric();
        if end > i + 1 && tail_ok && is_phone(chars, i, &chars[i..end]) {
            spans.push((i, end));
            i = end;
        } else {
            i += 1;
        }
    }
    spans
}

const SIGN_OFFS: [&str; 19] = [
    "best", "best regards", "regards", "kind regards", "warm regards", "best wishes", "sincerely",
    "yours sincerely", "yours faithfully", "yours truly", "thanks", "thank you", "many thanks",
    "thanks and regards", "cheers", "respectfully", "with thanks", "all the best", "warmly",
];

const NAME_LABELS: [&str; 12] = [
    "recruiter", "hiring manager", "contact", "posted by", "sent by", "recruiting contact",
    "talent partner", "reach out to", "point of contact", "your recruiter", "hiring contact", "from",
];

/// One token of a name: a capital, then letters, `-`, `'` (or a typographic apostrophe), or an
/// initial with a dot.
fn name_token(tok: &str) -> bool {
    let mut chars = tok.chars();
    let Some(first) = chars.next() else { return false };
    // A capital, or a letter of a script with no case (a Chinese or Korean name has no capital to
    // find, and a name that cannot be found is a name that stays).
    if !(first.is_uppercase() || (first.is_alphabetic() && !first.is_lowercase())) {
        return false;
    }
    let rest: Vec<char> = chars.collect();
    if rest.is_empty() {
        return false;
    }
    if rest == ['.'] {
        return true;
    }
    rest.iter().all(|c| c.is_alphabetic() || matches!(c, '-' | '\'' | '\u{2019}'))
}

/// Whether `s` (trimmed) is one to four name tokens, and nothing else.
fn name_like(s: &str) -> bool {
    let toks: Vec<&str> = s.split_whitespace().collect();
    !toks.is_empty() && toks.len() <= MAX_NAME_TOKENS && toks.iter().all(|t| name_token(t))
}

fn sign_off_key(line: &str) -> String {
    line.trim()
        .trim_end_matches([',', '.', '!', ':', '-'])
        .trim()
        .to_lowercase()
}

/// The byte range within `line` of a name to replace, if the line carries one on its own
/// (`Jane Doe`), after a sign-off (`Regards, Jane Doe`) or after a label (`Recruiter: Jane Doe`).
fn inline_name(line: &str) -> Option<(usize, usize)> {
    let lower = line.to_lowercase();
    let lead = line.len() - line.trim_start().len();
    // `to_lowercase` can change byte lengths; positions are only taken when it did not.
    if lower.len() != line.len() {
        return None;
    }
    for phrase in SIGN_OFFS.iter().chain(NAME_LABELS.iter()) {
        if !lower.get(lead..)?.starts_with(phrase) {
            continue;
        }
        let tail = line.get(lead + phrase.len()..)?;
        // A comma, colon or dash must follow: "Best of luck", "Best Practices" and "Contact Us"
        // are not sign-offs and labels.
        // ("Posted by", "Sent by" and "Reach out to" take the name directly.)
        let sep = tail.trim_start_matches(' ');
        let direct = phrase.ends_with(" by") || phrase.ends_with(" to");
        if !direct && !sep.starts_with([',', ':', '-', '\u{2013}', '\u{2014}']) {
            continue;
        }
        if direct && tail.len() == sep.len() {
            continue;
        }
        let name = sep
            .trim_start_matches([',', ':', '-', '\u{2013}', '\u{2014}', ' '])
            .trim_end();
        if name_like(name) {
            // `name` is a suffix of the line less its trailing whitespace.
            let end = line.trim_end().len();
            return Some((end - name.len(), end));
        }
    }
    None
}

/// Replace the signature-block names of `text`. Returns the text and how many names went.
fn redact_names(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len());
    let mut count = 0usize;
    let mut after_sign_off = false;
    for raw in text.split_inclusive('\n') {
        let content = raw.trim_end_matches(['\n', '\r']);
        let ending = &raw[content.len()..];
        let trimmed = content.trim();
        let mut line = content.to_owned();
        if trimmed.is_empty() {
            // Blank lines between a sign-off and the name do not end the block.
            out.push_str(&line);
            out.push_str(ending);
            continue;
        }
        if let Some((a, b)) = inline_name(content) {
            line.replace_range(a..b, NAME_PLACEHOLDER);
            count += 1;
            after_sign_off = false;
        } else if after_sign_off && name_like(trimmed) {
            let lead = content.len() - content.trim_start().len();
            line.replace_range(lead..lead + trimmed.len(), NAME_PLACEHOLDER);
            count += 1;
            after_sign_off = false;
        } else {
            after_sign_off = SIGN_OFFS.contains(&sign_off_key(trimmed).as_str());
        }
        out.push_str(&line);
        out.push_str(ending);
    }
    (out, count)
}

fn replace_spans(chars: &[char], spans: &[(usize, usize)], placeholder: &str) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut at = 0usize;
    for (s, e) in spans {
        out.extend(&chars[at..*s]);
        out.push_str(placeholder);
        at = *e;
    }
    out.extend(&chars[at..]);
    out
}

/// Remove emails, phone numbers and signature-block names.
pub fn redact(text: &str) -> Redacted {
    let chars: Vec<char> = text.chars().collect();
    let emails = email_spans(&chars);
    let after_email = replace_spans(&chars, &emails, EMAIL_PLACEHOLDER);
    let chars: Vec<char> = after_email.chars().collect();
    let phones = phone_spans(&chars);
    let after_phone = replace_spans(&chars, &phones, PHONE_PLACEHOLDER);
    let (text, names) = redact_names(&after_phone);
    Redacted {
        text,
        emails: emails.len(),
        phones: phones.len(),
        names,
    }
}

/// What a text still carries, by kind, read more strictly than [`redact`] writes: any at sign
/// (`@`, `＠`, `﹫`) counts as an email whether or not it parsed as one.
pub fn residual(text: &str) -> Vec<&'static str> {
    let chars: Vec<char> = text.chars().collect();
    let mut kinds = Vec::new();
    if chars.iter().any(|c| AT_SIGNS.contains(c)) || !email_spans(&chars).is_empty() {
        kinds.push("email");
    }
    if !phone_spans(&chars).is_empty() {
        kinds.push("phone");
    }
    let (_, names) = redact_names(text);
    if names > 0 {
        kinds.push("name");
    }
    kinds
}

/// Words of `text` that are not placeholders.
pub fn content_words(text: &str) -> usize {
    let stripped = text
        .replace(EMAIL_PLACEHOLDER, " ")
        .replace(PHONE_PLACEHOLDER, " ")
        .replace(NAME_PLACEHOLDER, " ");
    pyunicode::words(&pyunicode::lower(&stripped)).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red(s: &str) -> String {
        redact(s).text
    }

    #[test]
    fn plain_unicode_and_obfuscated_emails_are_removed() {
        assert_eq!(red("write to jane.doe+jobs@example.co.uk now"), "write to [EMAIL] now");
        assert_eq!(red("mail josé@exämple.com today"), "mail [EMAIL] today");
        assert_eq!(red("mail jane\u{ff20}example.com"), "mail [EMAIL]");
        assert_eq!(red("jane [at] example [dot] com"), "[EMAIL]");
        assert_eq!(red("jane (AT) example (dot) org please"), "[EMAIL] please");
        assert_eq!(red("jane at example dot com"), "[EMAIL]");
        assert_eq!(red("jane{ at }example.com"), "[EMAIL]");
    }

    #[test]
    fn ordinary_prose_with_at_is_not_an_email() {
        let s = "we are looking at remote roles and are at the heart of the work";
        assert_eq!(red(s), s);
    }

    #[test]
    fn international_and_national_phone_numbers_are_removed() {
        for s in [
            "+44 20 7946 0958",
            "+1 (415) 555-1234",
            "(415) 555-1234",
            "415-555-1234",
            "07911 123456",
            "4155551234",
            "+91 98765 43210",
            "+4915123456789",
        ] {
            assert_eq!(red(&format!("call {s} today")), "call [PHONE] today", "{s}");
        }
        assert_eq!(red("\u{ff14}\u{ff11}\u{ff15}-\u{ff15}\u{ff15}\u{ff15}-\u{ff11}\u{ff12}\u{ff13}\u{ff14}"), "[PHONE]");
        assert_eq!(red("\u{966}\u{967}\u{968}\u{969}\u{96a}\u{96b}\u{96c}\u{96d}\u{96e}\u{96f}"), "[PHONE]");
    }

    #[test]
    fn years_dates_ranges_and_versions_are_kept() {
        for s in [
            "from 2019-2023 we grew",
            "on 2026-10-21 we met",
            "on 10/21/2026 we met",
            "pay 100000-150000 per year",
            "salary $120,000 - $150,000",
            "Python 3.10.12 required",
            "5+ years of experience",
            "job id 12345678",
        ] {
            assert_eq!(red(s), s, "{s}");
        }
    }

    #[test]
    fn a_keyword_makes_a_short_number_a_phone() {
        assert_eq!(red("phone 555-1234 ext 9"), "phone [PHONE] ext 9");
    }

    #[test]
    fn the_name_after_a_sign_off_is_removed() {
        assert_eq!(
            red("Please apply.\n\nBest regards,\nJane Doe\nSenior Recruiter\nAcme"),
            "Please apply.\n\nBest regards,\n[NAME]\nSenior Recruiter\nAcme"
        );
        assert_eq!(red("Thanks,\n\nJos\u{e9} O'Neil-Smith\nTalent"), "Thanks,\n\n[NAME]\nTalent");
        assert_eq!(red("Regards, Jane Doe"), "Regards, [NAME]");
        assert_eq!(red("Recruiter: Jane Doe"), "Recruiter: [NAME]");
        assert_eq!(red("Posted by J. Smith"), "Posted by [NAME]");
        assert_eq!(red("Hiring manager - \u{5f20}\u{4f1f}"), "Hiring manager - [NAME]");
    }

    #[test]
    fn a_sentence_that_starts_with_a_sign_off_word_is_kept() {
        for s in ["Best of luck to every applicant", "Thanks for your interest in this role", "Contact us for details"] {
            assert_eq!(red(s), s);
        }
    }

    #[test]
    fn redaction_is_idempotent_and_leaves_nothing_behind() {
        let s = "Reach Jane at jane [at] acme [dot] com or +1 (415) 555-1234.\nBest,\nJane Doe";
        let once = redact(s);
        assert_eq!((once.emails, once.phones, once.names), (1, 1, 1));
        assert_eq!(redact(&once.text).text, once.text);
        assert!(residual(&once.text).is_empty(), "{:?}", once.text);
    }

    #[test]
    fn a_stray_at_sign_is_residual_even_when_it_parsed_as_nothing() {
        assert_eq!(residual("contact user@localhost"), vec!["email"]);
        assert_eq!(residual("ping \u{fe6b}jane"), vec!["email"]);
        assert!(residual("a perfectly clean posting about platform engineering").is_empty());
    }

    #[test]
    fn content_words_ignore_placeholders() {
        assert_eq!(content_words("[EMAIL] [PHONE] [NAME]"), 0);
        assert_eq!(content_words("one two [NAME] three"), 3);
    }
}
