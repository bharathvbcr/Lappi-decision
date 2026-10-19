//! The job-posting triage questions, synthesised by rule (see [`crate::synth`]).
//!
//! **What it mirrors.** The caller is Vani (`internal/decide/jdtriage`). Its `fixture.json`
//! (version 2, copied byte for byte to `fixtures/caller/career-jd-triage-fixture.json` and pinned
//! by sha256) is the one definition of the four families' tasks, exact questions, slot names,
//! options and context layout. Lappi's decision-pool driver carries one choice slot per example
//! (`synth::Draft`, `decisions::Candidate`), so the owner's triage question is asked as four
//! single-slot families, each with its own question:
//!
//! - `career.jd_role_fit` (slot `role_fit`), `career.jd_sponsorship` (`sponsorship`),
//!   `career.jd_work_mode` (`work_mode`) and `career.jd_seniority` (`seniority`).
//!
//! **Bare context.** Vani sends the question and the context separately, and the context is
//! [`render`]'s output and nothing else. So these are `decisions::BARE_CONTEXT_FAMILIES`: a row's
//! context is the rendered posting byte for byte, its `question` is empty, and the request's own
//! question is the family's description in `qd_data.sources.DECISION_FAMILIES`, which must be the
//! fixture's string. Options are the fixture's `value`s, in the fixture's order, which is the order
//! Vani sends them in (`synth::CALLER_ORDER_FAMILIES`).
//!
//! **The target header.** The context begins with `Target roles: <summary>` when the caller has a
//! target-role summary. For `career.jd_role_fit` the label is a function of (summary, posting), so
//! the header is part of the evidence: each row draws a target profile, then a role whose label
//! under that profile is the template's class ([`role_fit_label`]). An empty summary is the rule
//! the question states: judge against early-career software, ML or systems engineering roles
//! (profile `default`), and about one row in six has no header. For the other three families the
//! label does not depend on the summary, so a header is drawn per row from the row's own
//! [`Scope`], independent of the template and the label: both with and without a header occur in
//! every class, and the header cannot predict a label.
//!
//! **Labelling convention for level.** The posting's stated requirement beats its title. A
//! "Senior" title over a graduate programme with "0 to 1 years" is `intern_or_new_grad` (and, in
//! role fit, a level-appropriate role); an "Associate" title over "8+ years" and design
//! leadership is `senior` (and `mismatch_seniority` when the work matches the target roles).
//!
//! **Hard cases written in on purpose.**
//! - Sponsorship named only to deny it, including a company that sponsors hackathons and says the
//!   role has no visa sponsorship; relocation money with no visa support (`no_sponsorship`) beside
//!   relocation money with no word on visas (`not_stated`); a company that sponsors events and
//!   says nothing about visas (`not_stated`).
//! - "Remote within the US" (`remote`) beside remote Monday and Friday with three days in the
//!   office (`hybrid`); an annual offsite does not make a remote role hybrid; an office address,
//!   a headquarters town and core hours with no in-office rule are `not_stated`.
//! - Senior titles over graduate duties and associate titles over senior duties, a team lead with
//!   direct reports (`manager`) and a senior staff individual contributor (`staff_plus`).
//!
//! **Privacy.** Every company is invented, no person is named, no posting text is a real
//! employer's, and no profile value is read.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::decisions::{self, Gold};
use crate::sha256::sha256_hex;
use crate::synth::{Config, Draft, Generated, Scope, fill};

pub const ROLE_FIT_FAMILY: &str = "career.jd_role_fit";
pub const SPONSORSHIP_FAMILY: &str = "career.jd_sponsorship";
pub const WORK_MODE_FAMILY: &str = "career.jd_work_mode";
pub const SENIORITY_FAMILY: &str = "career.jd_seniority";
/// The four families, in the fixture's order.
pub const FAMILIES: [&str; 4] = [
    ROLE_FIT_FAMILY,
    SPONSORSHIP_FAMILY,
    WORK_MODE_FAMILY,
    SENIORITY_FAMILY,
];
pub const ROLE_FIT_SLOT: &str = "role_fit";
pub const SPONSORSHIP_SLOT: &str = "sponsorship";
pub const WORK_MODE_SLOT: &str = "work_mode";
pub const SENIORITY_SLOT: &str = "seniority";

/// Vani's `fixture.json`, version 2, byte for byte.
pub const FIXTURE_BYTES: &[u8] =
    include_bytes!("../../../fixtures/caller/career-jd-triage-fixture.json");
/// Its sha256; Vani pins the same digest on its side.
pub const FIXTURE_SHA256: &str = "0cf04ad791d67fb04bf4e1c5be9b3206f61cf356dddf0a3054c9a8b5a13be47d";
pub const FIXTURE_VERSION: u64 = 2;
/// `jdtriage.MaxContextBytes`.
pub const MAX_CONTEXT_BYTES: usize = 24_576;
/// `jdtriage.MaxTargetSummaryBytes`.
pub const MAX_SUMMARY_BYTES: usize = 1_024;

/// One family as the fixture states it.
#[derive(Clone, Debug)]
pub struct Spec {
    pub task: String,
    pub question: String,
    pub slot: String,
    /// The options' `value`s, in the fixture's order.
    pub options: Vec<String>,
}

/// The fixture's four families, after checking its digest, version, context layout and caps
/// against what [`render`] implements. Any disagreement is a refusal, never a quiet fallback: a
/// changed fixture is a contract change that needs a new generator and a retrain.
pub fn fixture() -> Result<Vec<Spec>, String> {
    let sha = sha256_hex(FIXTURE_BYTES);
    if sha != FIXTURE_SHA256 {
        return Err(format!(
            "the career fixture's sha256 is {sha}, not the pinned {FIXTURE_SHA256}"
        ));
    }
    let v: Value = serde_json::from_slice(FIXTURE_BYTES).map_err(|e| format!("fixture: {e}"))?;
    if decisions::get_u64(&v, "fixture_version", "fixture")? != FIXTURE_VERSION {
        return Err(format!("fixture: fixture_version is not {FIXTURE_VERSION}"));
    }
    let ctx = decisions::get(&v, "context", "fixture")?;
    let layout = "{target}Title: {title}\nLocation: {location}\n\n{description}";
    for (key, want) in [
        ("layout", layout),
        ("target", "Target roles: {summary}\n\n"),
        ("target_when_empty", ""),
    ] {
        if decisions::get_str(ctx, key, "fixture.context")? != want {
            return Err(format!(
                "fixture.context.{key} is not what render() implements"
            ));
        }
    }
    for (key, want) in [
        ("max_bytes", MAX_CONTEXT_BYTES),
        ("max_summary_bytes", MAX_SUMMARY_BYTES),
    ] {
        if decisions::get_u64(ctx, key, "fixture.context")? as usize != want {
            return Err(format!("fixture.context.{key} is not {want}"));
        }
    }
    let families = decisions::get(&v, "families", "fixture")?
        .as_array()
        .ok_or("fixture: families is not a list")?;
    let slots = [
        ROLE_FIT_SLOT,
        SPONSORSHIP_SLOT,
        WORK_MODE_SLOT,
        SENIORITY_SLOT,
    ];
    if families.len() != FAMILIES.len() {
        return Err(format!(
            "fixture: {} families, expected {}",
            families.len(),
            FAMILIES.len()
        ));
    }
    let mut out = Vec::new();
    for (i, f) in families.iter().enumerate() {
        let task = decisions::get_str(f, "task", "fixture.families")?;
        let slot = decisions::get(f, "slot", "fixture.families")?;
        let slot_name = decisions::get_str(slot, "name", "fixture.families.slot")?;
        if task != FAMILIES[i] || slot_name != slots[i] {
            return Err(format!(
                "fixture: family {i} is {task:?}/{slot_name:?}, expected {:?}/{:?}",
                FAMILIES[i], slots[i]
            ));
        }
        let options = decisions::get(slot, "options", "fixture.families.slot")?
            .as_array()
            .ok_or("fixture: options is not a list")?
            .iter()
            .map(|o| {
                decisions::get_str(o, "value", "fixture.families.slot.options").map(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(Spec {
            task: task.to_owned(),
            question: decisions::get_str(f, "question", "fixture.families")?.to_owned(),
            slot: slot_name.to_owned(),
            options,
        });
    }
    Ok(out)
}

/// `jdtriage.NormalizeSummary`: whitespace runs to one space, both ends trimmed, and a summary
/// over [`MAX_SUMMARY_BYTES`] refused.
pub fn normalize_summary(s: &str) -> Result<String, String> {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.len() > MAX_SUMMARY_BYTES {
        return Err(format!(
            "target summary is {} bytes, max {MAX_SUMMARY_BYTES}",
            s.len()
        ));
    }
    Ok(s)
}

/// `jdtriage.Render`: the context Vani sends, cut to [`MAX_CONTEXT_BYTES`] at a rune boundary.
/// `summary` must already be normalized; the flag says whether the cut happened.
pub fn render(title: &str, location: &str, description: &str, summary: &str) -> (String, bool) {
    let target = if summary.is_empty() {
        String::new()
    } else {
        format!("Target roles: {summary}\n\n")
    };
    let full = format!("{target}Title: {title}\nLocation: {location}\n\n{description}");
    if full.len() <= MAX_CONTEXT_BYTES {
        return (full, false);
    }
    let mut cut = MAX_CONTEXT_BYTES;
    while cut > 0 && !full.is_char_boundary(cut) {
        cut -= 1;
    }
    (full[..cut].to_owned(), true)
}

// ---------------------------------------------------------------------------------------------
// Posting templates (sponsorship, work mode, seniority).

/// One posting template: an id, a title, a location and a description, with `{placeholders}`.
pub(crate) struct Tpl {
    pub(crate) id: &'static str,
    title: &'static str,
    location: &'static str,
    body: &'static str,
}

const fn t(
    id: &'static str,
    title: &'static str,
    location: &'static str,
    body: &'static str,
) -> Tpl {
    Tpl {
        id,
        title,
        location,
        body,
    }
}

/// `(option, its templates)`. A template belongs to one option and its wording is its own.
pub(crate) type Table = &'static [(&'static str, &'static [Tpl])];

pub(crate) const SPONSORSHIP: Table = &[
    (
        "sponsors_visa",
        &[
            t(
                "visa-h1b",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for the {team} team, which builds {product}.\n\nWhat you will do:\n- Design and ship {tech} services\n- Review code and improve reliability\n\nVisa support: we sponsor H-1B and other work visas for qualified candidates, and our immigration counsel handles the filings at no cost to you.\n\nBenefits include health cover and a learning budget.",
            ),
            t(
                "visa-mobility",
                "Backend Engineer",
                "{city}",
                "Join {company} and help us build {product}.\n\nYou will own {tech} services end to end, from design through on-call.\n\nMoving from abroad? We provide visa sponsorship, a relocation package of ${amount} and a dedicated global mobility partner who guides you through the permit process.",
            ),
            t(
                "visa-opt",
                "New Graduate Engineer",
                "{city}",
                "{company}'s {team} team is growing, and we hire new graduates every year.\n\nStudents on F-1 OPT or STEM OPT are welcome to apply, and we will sponsor an H-1B petition for the right hire.\n\nYou will work in {tech} alongside a mentor on {product}.",
            ),
            t(
                "visa-uk",
                "Data Engineer",
                "London, UK",
                "{company} is looking for a data engineer to work on {product}.\n\nWe hold a UK Skilled Worker sponsor licence, so we can sponsor your visa if you need one, and we cover the application fees and the immigration health surcharge.\n\nStack: {tech}, SQL and a modern warehouse.",
            ),
            t(
                "visa-tn-o1",
                "Platform Engineer",
                "{city}",
                "About the role\nAs a platform engineer at {company} you will keep {product} fast and dependable.\n\nImmigration: we sponsor TN, H-1B and O-1 visas and green cards for employees who need them, and we never ask you to cover legal fees.",
            ),
        ],
    ),
    (
        "no_sponsorship",
        &[
            t(
                "deny-now-future",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for the {team} team, which builds {product}.\n\nWhat you will do:\n- Build {tech} services\n- Join the on-call rotation\n\nCandidates must be authorized to work in the United States. We are unable to sponsor work visas, now or in the future.",
            ),
            t(
                "deny-event-sponsor",
                "Backend Developer",
                "{city}",
                "{company} proudly sponsors local hackathons and a student robotics league, and we love giving back.\n\nHowever, this position is not eligible for visa sponsorship, so applicants must already hold the right to work in the country.\n\nYou will write {tech} for {product}.",
            ),
            t(
                "deny-right-to-work",
                "Site Reliability Engineer",
                "{city}",
                "Role: Site Reliability Engineer, {team}.\n\nYou must already have the legal right to work in the country where the role is based. This role does not offer immigration support.\n\nDay to day you will improve deploys and observability for {product}.",
            ),
            t(
                "deny-opt",
                "Junior Software Engineer",
                "{city}",
                "We are hiring a junior engineer for {product}.\n\nPlease note that we cannot offer visa sponsorship or OPT-based employment at this time; applicants who will need sponsorship now or later will not be considered.\n\nYou will pair with senior engineers on {tech} code.",
            ),
            t(
                "deny-boilerplate",
                "Software Engineer II",
                "{city}",
                "{company} is an equal opportunity employer.\n\nResponsibilities: build and run {tech} services for {product}.\n\nRequirements: applicants must be able to work for any employer in the United States without current or future sponsorship.",
            ),
            t(
                "deny-relocation",
                "Full-Stack Engineer",
                "{city}",
                "Come build {product} with the {team} team.\n\nWe offer relocation assistance of ${amount} to candidates who are already authorized to work in the country; visa sponsorship is not provided.",
            ),
        ],
    ),
    (
        "citizenship_or_clearance_required",
        &[
            t(
                "us-secret",
                "Software Engineer, {team}",
                "{city}",
                "{company} builds software for defense customers, including {product}.\n\nYou will write {tech} for mission systems.\n\nRequirements:\n- U.S. citizenship is required\n- Ability to obtain and maintain a Secret security clearance",
            ),
            t(
                "ts-sci",
                "Systems Engineer",
                "{city}",
                "Our {team} team supports {product} for an intelligence-community customer.\n\nRequired: an active TS/SCI clearance with polygraph. Candidates without a current clearance cannot be considered.",
            ),
            t(
                "itar",
                "Embedded Software Engineer",
                "{city}",
                "{company} designs flight hardware, and you will write {tech} firmware for it.\n\nBecause the work is subject to export-control rules (ITAR), applicants must be U.S. citizens or lawful permanent residents.",
            ),
            t(
                "uk-sc",
                "Software Developer",
                "Manchester, UK",
                "We are recruiting a developer to work on {product} for a government client.\n\nThe post requires British citizenship, and you must be eligible for SC security clearance.",
            ),
            t(
                "canada-pr",
                "Backend Developer",
                "Ottawa, ON",
                "{company} delivers {product} to federal agencies.\n\nDue to federal contract terms, this position is open only to Canadian citizens or permanent residents, and you must be able to obtain Reliability Status.",
            ),
        ],
    ),
    (
        "not_stated",
        &[
            t(
                "relocation-only",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for {product}.\n\nYou will build {tech} services and review code.\n\nIf you are moving to {city} for the job we offer a relocation stipend of ${amount} and two weeks of temporary housing.",
            ),
            t(
                "sponsors-events",
                "Backend Engineer",
                "{city}",
                "{company} sponsors regional hackathons and a youth coding league, and our engineers volunteer as mentors.\n\nIn this role you will own {tech} services for {product}, from design to on-call.",
            ),
            t(
                "eeo-citizenship",
                "Data Engineer",
                "{city}",
                "{company} is an equal opportunity employer. We do not discriminate on the basis of race, religion, citizenship status, age, disability or gender identity.\n\nYou will build pipelines for {product} using {tech} and SQL.",
            ),
            t(
                "background-check",
                "Platform Engineer",
                "{city}",
                "Join the {team} team at {company}.\n\nResponsibilities: run and improve the {tech} platform behind {product}.\n\nOffers are contingent on a background check and reference checks, as permitted by law.",
            ),
            t(
                "distributed-team",
                "Machine Learning Engineer",
                "{city}",
                "{company} is a distributed company with colleagues across nine countries and many time zones.\n\nYou will train and evaluate models for {product} and ship them with {tech}.\n\nBenefits: health cover, a learning budget and two weeks of company holidays.",
            ),
            t(
                "plain",
                "Software Engineer",
                "{city}",
                "{company} builds {product}, and the {team} team needs another engineer.\n\nYou will write {tech}, review code and take part in the on-call rotation.\n\nWe offer health cover, a learning budget and 25 days of paid leave.",
            ),
        ],
    ),
];

pub(crate) const WORK_MODE: Table = &[
    (
        "remote",
        &[
            t(
                "fully-remote",
                "Software Engineer, {team}",
                "Remote - US",
                "{company} is hiring a software engineer for {product}.\n\nThis is a fully remote role. Work from anywhere you have a reliable connection and a quiet place to focus.\n\nYou will build {tech} services and review code.",
            ),
            t(
                "remote-us-only",
                "Backend Engineer",
                "Remote (US)",
                "About the role\nThe {team} team at {company} is hiring a backend engineer.\n\nThis position is remote within the United States: you can live in any state where we can legally employ you and you will never be required to visit an office.\n\nYou will work on {product} in {tech}.",
            ),
            t(
                "remote-first",
                "Data Engineer",
                "Remote - Europe",
                "{company} is remote-first and has no headquarters. Every role, including this one, is done from home or from a coworking space of your choice, and we cover the coworking fees.\n\nYou will build pipelines for {product}.",
            ),
            t(
                "remote-offsite",
                "Platform Engineer",
                "{city}",
                "Platform engineer, {team}.\n\nYou can work entirely from home; there is no requirement to come into an office. The team meets in person once a year for a week-long offsite.\n\nYou will keep {product} reliable with {tech}.",
            ),
            t(
                "remote-async",
                "Site Reliability Engineer",
                "Remote",
                "Our team works asynchronously across time zones and the role is 100% remote.\n\nResponsibilities: on-call rotation, capacity planning and incident reviews for {product}.",
            ),
        ],
    ),
    (
        "hybrid",
        &[
            t(
                "three-days",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for {product}.\n\nThis is a hybrid role: three days a week in our {city} office and two days remote.\n\nYou will build {tech} services and review code.",
            ),
            t(
                "remote-mon-fri",
                "Backend Engineer",
                "{city}",
                "About the role\nThe {team} team is hiring a backend engineer for {product}.\n\nSchedule: remote on Monday and Friday, in the office Tuesday to Thursday.\n\nStack: {tech} and Postgres.",
            ),
            t(
                "twice-week",
                "Data Engineer",
                "{city}",
                "We are looking for a data engineer.\n\nWe ask everyone to come into the {city} office twice a week for planning and pairing; the rest of the week you work from wherever suits you.\n\nYou will build pipelines for {product}.",
            ),
            t(
                "hybrid-anchor",
                "Machine Learning Engineer",
                "{city} (Hybrid)",
                "Hybrid role based in {city}. Anchor days are Tuesday and Wednesday in the office, and the other days are flexible.\n\nYou will train and ship models for {product} using {tech}.",
            ),
            t(
                "remote-friendly-commute",
                "Platform Engineer",
                "{city}",
                "This is a remote-friendly role for candidates who live within 40 miles of our {city} office. You will join the team in person every Thursday and Friday and work from home the other days.\n\nYou will keep {product} reliable.",
            ),
        ],
    ),
    (
        "onsite",
        &[
            t(
                "five-days",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for {product}.\n\nThis is an on-site role. You will work from our {city} office five days a week.\n\nYou will build {tech} services and review code.",
            ),
            t(
                "no-remote",
                "Embedded Software Engineer",
                "{city}",
                "About the role\nYou will bring up new boards and test them on the bench.\n\nNo remote work: the job requires you to be in our {city} lab every day because the hardware cannot leave the building.",
            ),
            t(
                "plant-shifts",
                "Controls Engineer",
                "{city}",
                "{company} runs a packaging plant and you will maintain the control software that runs its lines.\n\nThe role is fully on-site at the plant, with day shifts and an occasional weekend call-out.",
            ),
            t(
                "in-person-culture",
                "Backend Engineer",
                "{city}",
                "We believe in building together in person. Everyone on the {team} team works full time from the {city} office, and this role is no exception.\n\nYou will build {product} in {tech}.",
            ),
            t(
                "not-remote",
                "Data Engineer",
                "{city}",
                "Data engineer, {team}.\n\nThis position is not remote and is not hybrid: it is based full time in the office, and applicants who want to work from home will not be considered.\n\nYou will build pipelines for {product}.",
            ),
        ],
    ),
    (
        "not_stated",
        &[
            t(
                "address-only",
                "Software Engineer, {team}",
                "{city}",
                "{company} is hiring a software engineer for {product}.\n\nOur office is at {street}, {city}, a short walk from the main station.\n\nYou will build {tech} services and review code.",
            ),
            t(
                "hq-located",
                "Backend Engineer",
                "{city}",
                "{company} is headquartered in {city} and was founded in 2012. We build {product} for customers in forty countries.\n\nAs a backend engineer you will own {tech} services from design to on-call.",
            ),
            t(
                "core-hours",
                "Data Engineer",
                "{city}",
                "Data engineer, {team}.\n\nThe team overlaps on core hours from 10:00 to 15:00 Eastern and everything else is flexible. You will build pipelines for {product} and keep the warehouse tidy.",
            ),
            t(
                "benefits-plain",
                "Platform Engineer",
                "{city}",
                "About the role\nYou will run and improve the {tech} platform behind {product}.\n\nBenefits: health and dental cover, a learning budget, 401(k) matching and 25 days of paid leave.",
            ),
            t(
                "team-events",
                "Machine Learning Engineer",
                "{city}",
                "Join {company} and train models for {product}.\n\nWe hold a team summit each autumn and a volunteering day each spring. You will use {tech} and a lot of evaluation data.",
            ),
        ],
    ),
];

pub(crate) const SENIORITY: Table = &[
    (
        "intern_or_new_grad",
        &[
            t(
                "intern-summer",
                "Software Engineering Intern (Summer {year})",
                "{city}",
                "{company} is hiring interns for summer {year}.\n\nThis 12-week internship places you on the {team} team, where you will ship a real feature in {tech} with a mentor. Open to students currently enrolled in a degree program.",
            ),
            t(
                "new-grad",
                "Software Engineer, New Graduate",
                "{city}",
                "Our new graduate program is for people finishing a degree within the next year.\n\nNo professional experience is required. You will join {team}, work on {product} in {tech}, and have a mentor for your first year.",
            ),
            t(
                "senior-title-grad",
                "Senior Software Engineer Trainee (University Graduate Program)",
                "{city}",
                "{company}'s graduate program uses a traditional title ladder, so graduates join as Senior Software Engineer Trainees.\n\nWho we want: recent graduates with 0 to 1 years of experience. You will rotate through three teams, work under a mentor and write {tech} for {product}.",
            ),
            t(
                "co-op",
                "Engineering Co-op (4 months)",
                "{city}",
                "Co-op students spend four months with the {team} team.\n\nYou must be enrolled in a university co-op program. You will fix bugs and add tests in {tech} for {product}.",
            ),
            t(
                "entry-level",
                "Entry-Level Software Engineer",
                "{city}",
                "We are hiring an entry-level engineer.\n\nRequirements: a bachelor's degree and 0 to 1 years of experience. We will teach you our {tech} stack and how we run {product}.",
            ),
        ],
    ),
    (
        "mid",
        &[
            t(
                "se2",
                "Software Engineer II",
                "{city}",
                "{company} is hiring a Software Engineer II for the {team} team.\n\nYou have 3+ years of professional experience, can take a feature from design to production on your own and are comfortable with {tech}.",
            ),
            t(
                "two-to-four",
                "Backend Engineer",
                "{city}",
                "About the role\nYou will build services for {product}.\n\nRequirements: 2 to 4 years of professional experience in {tech} or a similar language, and you are comfortable owning small projects.",
            ),
            t(
                "mid-level-data",
                "Mid-Level Data Engineer",
                "{city}",
                "We need a mid-level data engineer with 3 to 5 years of experience building pipelines. You will work with senior engineers on {product} and mentor interns.",
            ),
            t(
                "intermediate",
                "Intermediate Developer",
                "{city}",
                "{company}'s {team} team is looking for an intermediate developer with about four years of experience.\n\nYou will own features for {product}, work independently on well-defined problems and escalate the hard ones.",
            ),
            t(
                "mid-nomenclature",
                "Software Engineer, {team}",
                "{city}",
                "Role: Software Engineer, {team}.\n\nWe expect at least three years of production experience with {tech}. You will deliver features on your own, review peers' code and be mentored by our senior engineers.",
            ),
        ],
    ),
    (
        "senior",
        &[
            t(
                "senior-se",
                "Senior Software Engineer",
                "{city}",
                "{company} is hiring a Senior Software Engineer for {product}.\n\nYou bring 6+ years of experience, design services for scale and mentor the engineers around you.",
            ),
            t(
                "associate-8yrs",
                "Associate Platform Engineer",
                "{city}",
                "Despite the title, which comes from our consulting heritage, this role needs 8+ years of experience.\n\nYou will lead design reviews, set technical direction for the {team} team's services and mentor others. You write {tech} daily and have no direct reports.",
            ),
            t(
                "sr-5",
                "Sr. Backend Engineer",
                "{city}",
                "About the role\nYou will lead the design of backend systems for {product}.\n\nRequirements: 5+ years of experience shipping and operating production services.",
            ),
            t(
                "lead-ic",
                "Lead Engineer (individual contributor)",
                "{city}",
                "Lead the technical work of a small squad on {product}. This is a hands-on individual contributor role with no direct reports.\n\nWe look for 7+ years of experience and a record of mentoring.",
            ),
            t(
                "senior-data",
                "Senior Data Engineer",
                "{city}",
                "We are looking for a senior data engineer with at least five years of experience.\n\nYou will design the pipelines behind {product} and review the work of the team in {tech}.",
            ),
        ],
    ),
    (
        "staff_plus",
        &[
            t(
                "staff",
                "Staff Software Engineer",
                "{city}",
                "{company} is hiring a Staff Software Engineer.\n\nYou have 10+ years of experience, set technical strategy across several teams and drive projects that no single team owns, for {product}.",
            ),
            t(
                "principal",
                "Principal Engineer, {team}",
                "{city}",
                "As Principal Engineer you will define the architecture of {product} for the next three years, represent engineering in executive reviews and mentor staff engineers. Individual contributor role; 12+ years of experience.",
            ),
            t(
                "distinguished",
                "Distinguished Engineer",
                "{city}",
                "The Distinguished Engineer is {company}'s most senior individual contributor. You will shape our technical direction across all of {product} and advise the CTO, without managing a team.",
            ),
            t(
                "staff-ml",
                "Staff Machine Learning Engineer",
                "{city}",
                "We are hiring a Staff Machine Learning Engineer to lead modelling strategy for {product} across teams. 9+ years of experience; you will write code, set standards and unblock senior engineers.",
            ),
            t(
                "senior-staff",
                "Senior Staff Engineer",
                "{city}",
                "This senior staff role sets cross-organisation technical direction for the {team} area. You will lead through influence, not reports, and own the hardest {tech} design problems at {company}.",
            ),
        ],
    ),
    (
        "manager",
        &[
            t(
                "em",
                "Engineering Manager, {team}",
                "{city}",
                "You will manage a team of six to eight engineers on {product}: hiring, coaching, performance reviews and delivery.\n\nWe look for 5+ years of engineering experience and 2+ years managing engineers.",
            ),
            t(
                "director",
                "Director of Engineering",
                "{city}",
                "As Director of Engineering you will lead three managers and about twenty engineers, own the roadmap and budget for {product} and report to the CTO.",
            ),
            t(
                "head-of",
                "Head of Platform",
                "{city}",
                "The Head of Platform builds and leads the platform organisation at {company}: set the hiring plan, manage the managers, and be accountable for the reliability of {product}.",
            ),
            t(
                "team-lead-reports",
                "Software Team Lead",
                "{city}",
                "You will be the line manager for five engineers on the {team} team: one-on-ones, hiring, promotions and performance reviews. About a third of your time is still hands-on {tech}.",
            ),
            t(
                "senior-manager",
                "Senior Engineering Manager",
                "{city}",
                "You will manage engineering managers across two product areas, grow the team from 12 to 25 people and own delivery for {product}.",
            ),
        ],
    ),
];

// ---------------------------------------------------------------------------------------------
// Role fit: the label is a function of the target profile and the role.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// Early-career: a degree or up to a few years of experience.
    Entry,
    /// Far above early-career: principal, director or executive, with a decade or more required.
    Far,
}

/// A role in the catalogue. `side` is a second domain for a role that splits its time.
pub(crate) struct Role {
    pub(crate) title: &'static str,
    duties: &'static str,
    reqs: &'static str,
    pub(crate) domain: &'static str,
    pub(crate) side: Option<&'static str>,
    pub(crate) level: Level,
}

const fn role(
    title: &'static str,
    duties: &'static str,
    reqs: &'static str,
    domain: &'static str,
    level: Level,
) -> Role {
    Role {
        title,
        duties,
        reqs,
        domain,
        side: None,
        level,
    }
}

const fn split_role(
    title: &'static str,
    duties: &'static str,
    domain: &'static str,
    side: &'static str,
) -> Role {
    Role {
        title,
        duties,
        reqs: "up to three years of experience",
        domain,
        side: Some(side),
        level: Level::Entry,
    }
}

use Level::{Entry, Far};

pub(crate) const ROLES: &[Role] = &[
    role(
        "Software Engineer, Backend Services",
        "build and operate backend services behind our public API, write tests and join the on-call rotation",
        "a degree or equivalent experience and up to two years of professional experience",
        "backend",
        Entry,
    ),
    role(
        "Junior Backend Developer",
        "implement API endpoints, fix bugs and learn our deployment pipeline",
        "0 to 2 years of experience with a typed language",
        "backend",
        Entry,
    ),
    role(
        "Senior Software Engineer (New Graduate Programme)",
        "ship small backend features with a mentor; the programme uses a senior title ladder",
        "new graduates welcome and no prior professional experience required",
        "backend",
        Entry,
    ),
    role(
        "Principal Backend Engineer",
        "own the architecture of our service platform across all product lines",
        "12+ years of experience designing large distributed systems",
        "backend",
        Far,
    ),
    role(
        "Associate Backend Engineer",
        "set technical direction for the platform and lead design reviews for every team",
        "10+ years of experience; the associate title is a legacy of our consulting heritage",
        "backend",
        Far,
    ),
    role(
        "Frontend Engineer",
        "build accessible interfaces in TypeScript and contribute to the shared component library",
        "up to two years of experience building web interfaces",
        "frontend",
        Entry,
    ),
    role(
        "Associate UI Engineer",
        "turn designs into responsive pages, fix layout bugs and write component tests",
        "a degree or a strong portfolio and 0 to 1 years of experience",
        "frontend",
        Entry,
    ),
    role(
        "Principal Frontend Architect",
        "define the web architecture for every product surface",
        "15+ years of experience and a record of leading platform migrations",
        "frontend",
        Far,
    ),
    role(
        "Machine Learning Engineer, Inference",
        "optimise model-serving latency, profile GPU kernels and run evaluation suites for transformer inference",
        "0 to 3 years of experience, strong Python and some C++ or Rust",
        "ml",
        Entry,
    ),
    role(
        "Junior ML Engineer",
        "train and evaluate classifiers, build data loaders and run experiments under guidance",
        "a degree in a quantitative field or equivalent experience",
        "ml",
        Entry,
    ),
    role(
        "ML Engineer, Agents",
        "build tool-using agent workflows and the evaluation harnesses that measure them",
        "up to three years of experience shipping ML or LLM features",
        "ml",
        Entry,
    ),
    role(
        "Distinguished Machine Learning Scientist",
        "set the research direction for the company's modelling work",
        "15+ years of experience and a publication record",
        "ml",
        Far,
    ),
    role(
        "Systems Engineer, Storage",
        "work on storage-engine internals in Rust: caching, compaction and crash recovery",
        "0 to 3 years of experience with systems programming",
        "systems",
        Entry,
    ),
    role(
        "Site Reliability Engineer I",
        "automate deployments and improve observability for our compute clusters",
        "a degree or equivalent experience; early-career engineers encouraged",
        "systems",
        Entry,
    ),
    role(
        "Compiler Engineer (Early Career)",
        "implement optimisation passes and fix miscompiles in our toolchain",
        "new graduates and up to two years of experience",
        "systems",
        Entry,
    ),
    role(
        "Principal Systems Architect",
        "own the multi-year architecture of our infrastructure and kernel teams",
        "14+ years of experience in operating systems or distributed storage",
        "systems",
        Far,
    ),
    role(
        "Data Engineer",
        "build batch and streaming pipelines, and maintain the warehouse and model tables for analysts",
        "0 to 3 years of experience with SQL and Python",
        "data",
        Entry,
    ),
    role(
        "Analytics Engineer I",
        "write tested SQL models, document metrics and keep dashboards accurate",
        "up to two years of experience in analytics or data engineering",
        "data",
        Entry,
    ),
    role(
        "Head of Data Platform",
        "own the data platform strategy and architecture across the company",
        "12+ years of experience building data platforms",
        "data",
        Far,
    ),
    role(
        "Embedded Firmware Engineer",
        "write C firmware for motor controllers and bring up new boards on the bench",
        "0 to 3 years of experience with microcontrollers",
        "embedded",
        Entry,
    ),
    role(
        "Robotics Software Engineer I",
        "develop perception and control software for mobile robots and test it in simulation",
        "a degree in robotics, computer or electrical engineering; early-career welcome",
        "embedded",
        Entry,
    ),
    role(
        "Principal Robotics Architect",
        "define the software architecture for the whole robot fleet",
        "15+ years of experience shipping robots",
        "embedded",
        Far,
    ),
    role(
        "Security Engineer, Detection",
        "write detection rules, triage alerts and automate response playbooks in Python",
        "0 to 3 years of experience in security or software engineering",
        "security",
        Entry,
    ),
    role(
        "Application Security Engineer I",
        "review code for vulnerabilities, run scanners and help developers fix findings",
        "a degree or equivalent experience; early-career engineers welcome",
        "security",
        Entry,
    ),
    role(
        "Chief Information Security Officer",
        "set the security strategy for the company and report to the board",
        "15+ years of experience in security leadership",
        "security",
        Far,
    ),
    role(
        "Registered Nurse, Medical-Surgical Unit",
        "provide bedside care, administer medication and chart patient progress",
        "a nursing licence and up to two years of experience",
        "nursing",
        Entry,
    ),
    role(
        "Pastry Chef de Partie",
        "prepare laminated doughs and plated desserts for a 90-seat restaurant",
        "1 to 3 years in a professional kitchen",
        "culinary",
        Entry,
    ),
    role(
        "Staff Accountant",
        "close the month, reconcile accounts and prepare tax schedules",
        "a bachelor's degree in accounting and 0 to 2 years of experience",
        "accounting",
        Entry,
    ),
    role(
        "Territory Sales Representative",
        "call on hardware stores across the region and hit a quarterly quota",
        "1 to 3 years of field sales experience",
        "field_sales",
        Entry,
    ),
    role(
        "Warehouse Associate",
        "pick, pack and load pallets for outbound freight",
        "no experience required; forklift certification a plus",
        "warehouse",
        Entry,
    ),
    role(
        "Paralegal",
        "draft contracts, manage filings and support attorneys on a commercial litigation team",
        "a paralegal certificate and 0 to 2 years of experience",
        "legal",
        Entry,
    ),
    role(
        "Visual Designer",
        "create brand illustrations and marketing layouts",
        "a portfolio and up to two years of design experience",
        "design",
        Entry,
    ),
    role(
        "Chief Financial Officer",
        "lead finance, treasury and investor relations and report to the board",
        "18+ years of experience in corporate finance",
        "accounting",
        Far,
    ),
    role(
        "Director of Nursing",
        "lead nursing operations across three hospital sites and manage the budget",
        "15+ years of experience in clinical leadership",
        "nursing",
        Far,
    ),
    split_role(
        "Applied ML Engineer and Sales Engineer",
        "spend half your time training and evaluating models and half presenting demos and negotiating pilots with prospects",
        "ml",
        "field_sales",
    ),
    split_role(
        "Backend Developer and Billing Operator",
        "write backend services for invoicing and also run month-end billing reconciliation by hand",
        "backend",
        "accounting",
    ),
    split_role(
        "Data Engineer and Warehouse Inventory Lead",
        "build data pipelines and also supervise weekly stock counts on the warehouse floor",
        "data",
        "warehouse",
    ),
    split_role(
        "Firmware Engineer and Field Installer",
        "write firmware and spend half the week installing and servicing devices at customer sites",
        "embedded",
        "field_service",
    ),
    split_role(
        "Security Engineer and Compliance Bookkeeper",
        "write detection rules and also prepare audit filings and ledgers",
        "security",
        "accounting",
    ),
    split_role(
        "Frontend Engineer and Brand Designer",
        "build web interfaces and also produce illustrations and marketing layouts",
        "frontend",
        "design",
    ),
    split_role(
        "Systems Engineer and Facilities Coordinator",
        "maintain our compute clusters and also manage office leases and cleaning contracts",
        "systems",
        "facilities",
    ),
];

/// A target-role profile: what its summaries say, and which role domains are the target work
/// (`core`) and which neighbour it (`adjacent`). `default` has no summary: the question's rule
/// for an empty one is early-career software, ML or systems engineering.
pub(crate) struct Profile {
    pub(crate) id: &'static str,
    pub(crate) summaries: &'static [&'static str],
    core: &'static [&'static str],
    adjacent: &'static [&'static str],
}

pub(crate) const PROFILES: &[Profile] = &[
    Profile {
        id: "default",
        summaries: &[],
        core: &["backend", "frontend", "ml", "systems"],
        adjacent: &["data", "embedded", "security"],
    },
    Profile {
        id: "ml_systems",
        summaries: &[
            "ML systems / inference / agents engineering roles",
            "machine learning infrastructure, model serving and agent tooling",
        ],
        core: &["ml", "systems"],
        adjacent: &["backend", "data", "embedded"],
    },
    Profile {
        id: "data",
        summaries: &[
            "data engineering and analytics platform roles",
            "pipelines,  warehouse and\nanalytics engineering",
        ],
        core: &["data"],
        adjacent: &["backend", "ml", "systems"],
    },
    Profile {
        id: "embedded",
        summaries: &[
            "embedded firmware and robotics software roles",
            "firmware, real-time systems and robotics",
        ],
        core: &["embedded"],
        adjacent: &["systems", "ml"],
    },
    Profile {
        id: "frontend",
        summaries: &[
            "frontend and design-systems engineering roles",
            "web UI engineering, component libraries and accessibility",
        ],
        core: &["frontend"],
        adjacent: &["backend", "design"],
    },
    Profile {
        id: "security",
        summaries: &[
            "security engineering and detection roles",
            "application security, detection and incident response engineering",
        ],
        core: &["security"],
        adjacent: &["systems", "backend"],
    },
];

/// The `career.jd_role_fit` label of `role` under `profile`, or `None` for a pair whose label
/// the question does not settle (a role in a neighbouring field at the wrong level, or a split
/// role that touches no target work); the generator never drafts those.
///
/// - Work that is the target's own, at an early-career level: `strong_fit`; at a level far above
///   it: `mismatch_seniority`.
/// - Work in a neighbouring field at an early-career level: `adjacent`.
/// - Work in an unrelated field, at any level: `mismatch_domain`.
/// - A role splitting its time between target work and something else: `partial_fit`.
pub(crate) fn role_fit_label(profile: &Profile, role: &Role) -> Option<&'static str> {
    if let Some(side) = role.side {
        let core = [role.domain, side]
            .iter()
            .filter(|d| profile.core.contains(d))
            .count();
        return match core {
            1 => Some("partial_fit"),
            2 => Some("strong_fit"),
            _ => None,
        };
    }
    let core = profile.core.contains(&role.domain);
    let adjacent = profile.adjacent.contains(&role.domain);
    match (role.level, core, adjacent) {
        (Entry, true, _) => Some("strong_fit"),
        (Entry, false, true) => Some("adjacent"),
        (Far, true, _) => Some("mismatch_seniority"),
        (Far, false, true) => None,
        (_, false, false) => Some("mismatch_domain"),
    }
}

/// Posting layouts for role fit, `(id, title, body)`. A template is a layout under one option, so
/// each option has as many templates as there are layouts.
const LAYOUTS: [(&str, &str, &str); 5] = [
    (
        "plain",
        "{title}",
        "{company} builds {product}.\n\nIn this role you will {duties}.\n\nRequirements: {reqs}.\n\nWe offer health cover and a learning budget.",
    ),
    (
        "team",
        "{title}, {team}",
        "About the team\nThe {team} team at {company} works on {product}.\n\nWhat you will do\n- You will {duties}.\n\nWhat we need\n- {reqs}",
    ),
    (
        "opportunity",
        "{title} - {company}",
        "Here is the opportunity: {company} is hiring.\n\nDay to day you will {duties}.\n\nYou bring: {reqs}.",
    ),
    (
        "join",
        "{title}",
        "Join {company} to work on {product}.\n\nThe work: {duties}.\n\nExperience we expect: {reqs}.\n\nPerks: 25 days of paid leave and a home-office stipend.",
    ),
    (
        "summary",
        "{title}",
        "Role summary\n{company} / {team}\nDuties: {duties}.\nRequired: {reqs}.",
    ),
];

// ---------------------------------------------------------------------------------------------
// Fills.

const COMPANIES: [&str; 16] = [
    "Brindlewick Labs",
    "Tallowmere Systems",
    "Quillfeather Analytics",
    "Ostrander and Pike",
    "Halcyon Reach",
    "Marrowgate Robotics",
    "Fennimore Works",
    "Cobaltine Health",
    "Lowmoor Freight",
    "Penhallow Data",
    "Sundervale Cloud",
    "Tidewrack Security",
    "Aldermoor Software",
    "Pinegrove Mobility",
    "Wrenfield Networks",
    "Thistledown Energy",
];
const CITIES: [&str; 10] = [
    "Austin, TX",
    "Denver, CO",
    "Pittsburgh, PA",
    "Raleigh, NC",
    "Seattle, WA",
    "Chicago, IL",
    "Boston, MA",
    "Atlanta, GA",
    "Portland, OR",
    "Minneapolis, MN",
];
const STREETS: [&str; 6] = [
    "Harbour Row",
    "Alder Street",
    "Mill Lane",
    "Quarry Road",
    "Larch Avenue",
    "Foundry Street",
];
const TEAMS: [&str; 8] = [
    "Platform",
    "Payments",
    "Search",
    "Identity",
    "Ingestion",
    "Developer Experience",
    "Ledger",
    "Routing",
];
const TECH: [&str; 7] = [
    "Rust",
    "Go",
    "Kotlin",
    "TypeScript",
    "Python",
    "C++",
    "Swift",
];
const PRODUCTS: [&str; 8] = [
    "fleet telemetry",
    "clinic scheduling",
    "freight routing",
    "ad analytics",
    "grid forecasting",
    "document search",
    "payment reconciliation",
    "warehouse robotics",
];
const AMOUNTS: [&str; 4] = ["2,500", "4,000", "6,000", "8,500"];
const YEARS: [&str; 2] = ["2027", "2028"];

fn vars(s: &Scope) -> BTreeMap<&'static str, String> {
    let pick = |tag: &str, xs: &[&str]| (*s.pick(tag, xs)).to_owned();
    let mut v = BTreeMap::new();
    v.insert("company", pick("company", &COMPANIES));
    v.insert("city", pick("city", &CITIES));
    v.insert(
        "street",
        format!(
            "{} {}",
            100 + s.below("street-number", 900),
            pick("street", &STREETS)
        ),
    );
    v.insert("team", pick("team", &TEAMS));
    v.insert("tech", pick("tech", &TECH));
    v.insert("product", pick("product", &PRODUCTS));
    v.insert("amount", pick("amount", &AMOUNTS));
    v.insert("year", pick("year", &YEARS));
    v
}

/// Every non-empty target summary of every profile: what a caller's header can say, drawn for
/// the families whose label does not depend on it.
fn all_summaries() -> Vec<&'static str> {
    PROFILES
        .iter()
        .flat_map(|p| p.summaries.iter().copied())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Generation.

#[derive(Default)]
struct Counts {
    with_header: usize,
    without_header: usize,
}

struct Row<'a> {
    spec: &'a Spec,
    family: &'static str,
    slot: &'static str,
    template: String,
    class: &'a str,
}

/// Render one posting and append its draft. A context that [`render`] had to cut is refused: no
/// row of a pool is a truncated posting.
fn emit(
    out: &mut Vec<Draft>,
    row: &Row,
    title: &str,
    location: &str,
    description: &str,
    summary: &str,
) -> Result<(), String> {
    let (context, truncated) = render(title, location, description, summary);
    if truncated {
        return Err(format!(
            "{}: a drafted posting is over {MAX_CONTEXT_BYTES} bytes",
            row.template
        ));
    }
    let gold = row
        .spec
        .options
        .iter()
        .position(|o| o == row.class)
        .ok_or_else(|| {
            format!(
                "{}: class {:?} is not one of {}'s options {:?}",
                row.template, row.class, row.spec.task, row.spec.options
            )
        })?;
    out.push(Draft {
        family_id: row.family,
        stratum: format!("{}/{}", row.family, row.class),
        template: row.template.clone(),
        template_class: row.class.to_owned(),
        context,
        // A bare-context family: the request's question is the family's description.
        question: String::new(),
        slot_name: row.slot,
        options: row.spec.options.clone(),
        gold: Gold::Option(gold),
        fixed_options: true,
    });
    Ok(())
}

fn scope_for(cfg: &Config, family: &str, template: &str, r: usize) -> Scope {
    Scope::new(
        cfg.seed,
        &format!("career\u{1f}{family}\u{1f}{template}\u{1f}{r}"),
    )
}

/// Whether the row carries a target header, and its normalized summary: a coin from the row's
/// own scope, so the header is independent of the template and of the label.
fn optional_header(s: &Scope, summaries: &[&str]) -> Result<String, String> {
    if s.unit("target") < 0.5 {
        return Ok(String::new());
    }
    normalize_summary(s.pick("summary", summaries))
}

fn single_slot_family(
    cfg: &Config,
    spec: &Spec,
    family: &'static str,
    slot: &'static str,
    table: Table,
    counts: &mut Counts,
    out: &mut Vec<Draft>,
) -> Result<(), String> {
    let summaries = all_summaries();
    for (class, templates) in table {
        for tpl in *templates {
            let template = format!("{class}:{}", tpl.id);
            for r in 0..cfg.rows_per_template {
                let s = scope_for(cfg, family, &template, r);
                let v = vars(&s);
                let summary = optional_header(&s, &summaries)?;
                if summary.is_empty() {
                    counts.without_header += 1;
                } else {
                    counts.with_header += 1;
                }
                let row = Row {
                    spec,
                    family,
                    slot,
                    template: template.clone(),
                    class,
                };
                emit(
                    out,
                    &row,
                    &fill(tpl.title, &v)?,
                    &fill(tpl.location, &v)?,
                    &fill(tpl.body, &v)?,
                    &summary,
                )?;
            }
        }
    }
    Ok(())
}

fn role_fit_family(
    cfg: &Config,
    spec: &Spec,
    counts: &mut Counts,
    out: &mut Vec<Draft>,
) -> Result<(), String> {
    for class in &spec.options {
        for (layout, title_tpl, body_tpl) in LAYOUTS {
            let template = format!("{class}:{layout}");
            for r in 0..cfg.rows_per_template {
                let s = scope_for(cfg, ROLE_FIT_FAMILY, &template, r);
                // A profile whose roles include one with this label; the draw is over profiles,
                // so every profile (the empty summary included) meets every label.
                let order = s.shuffle("profile", &(0..PROFILES.len()).collect::<Vec<_>>());
                let (profile, role) = order
                    .iter()
                    .find_map(|&i| {
                        let candidates: Vec<&Role> = ROLES
                            .iter()
                            .filter(|r| role_fit_label(&PROFILES[i], r) == Some(class.as_str()))
                            .collect();
                        (!candidates.is_empty())
                            .then(|| (&PROFILES[i], *s.pick("role", &candidates)))
                    })
                    .ok_or_else(|| format!("no role has the label {class:?} under any profile"))?;
                let mut v = vars(&s);
                v.insert("title", role.title.to_owned());
                v.insert("duties", role.duties.to_owned());
                v.insert("reqs", role.reqs.to_owned());
                let summary = if profile.summaries.is_empty() {
                    String::new()
                } else {
                    normalize_summary(s.pick("summary", profile.summaries))?
                };
                if summary.is_empty() {
                    counts.without_header += 1;
                } else {
                    counts.with_header += 1;
                }
                let row = Row {
                    spec,
                    family: ROLE_FIT_FAMILY,
                    slot: ROLE_FIT_SLOT,
                    template: template.clone(),
                    class,
                };
                // The title is filled before the layout, so the title line is
                // "<role title><layout suffix>" and [`role_fit_gold`] can find the role again.
                emit(
                    out,
                    &row,
                    &fill(title_tpl, &v)?,
                    &fill("{city}", &v)?,
                    &fill(body_tpl, &v)?,
                    &summary,
                )?;
            }
        }
    }
    Ok(())
}

/// The four families' drafts, the fixture's questions and options in the manifest, and the
/// fixture's digest as an input.
pub fn generate(cfg: &Config) -> Result<Generated, String> {
    let specs = fixture()?;
    let mut drafts = Vec::new();
    let mut report = serde_json::Map::new();
    for spec in &specs {
        let mut counts = Counts::default();
        match spec.task.as_str() {
            ROLE_FIT_FAMILY => role_fit_family(cfg, spec, &mut counts, &mut drafts)?,
            SPONSORSHIP_FAMILY => single_slot_family(
                cfg,
                spec,
                SPONSORSHIP_FAMILY,
                SPONSORSHIP_SLOT,
                SPONSORSHIP,
                &mut counts,
                &mut drafts,
            )?,
            WORK_MODE_FAMILY => single_slot_family(
                cfg,
                spec,
                WORK_MODE_FAMILY,
                WORK_MODE_SLOT,
                WORK_MODE,
                &mut counts,
                &mut drafts,
            )?,
            SENIORITY_FAMILY => single_slot_family(
                cfg,
                spec,
                SENIORITY_FAMILY,
                SENIORITY_SLOT,
                SENIORITY,
                &mut counts,
                &mut drafts,
            )?,
            other => return Err(format!("fixture family {other:?} has no generator")),
        }
        report.insert(
            spec.task.clone(),
            json!({
                "question": spec.question, "slot": spec.slot, "options": spec.options,
                "rows_with_target_header": counts.with_header,
                "rows_without_target_header": counts.without_header,
            }),
        );
    }
    Ok(Generated {
        drafts,
        report: json!({
            "fixture_version": FIXTURE_VERSION,
            "fixture_sha256": FIXTURE_SHA256,
            "families": report,
            "role_fit_profiles": PROFILES.iter().map(|p| p.id).collect::<Vec<_>>(),
            "gold_rule": "a template belongs to one option; role fit draws a target profile and \
                          a role whose label under that profile is the template's option \
                          (role_fit_label)",
        }),
        inputs: [("career_fixture".to_owned(), sha256_hex(FIXTURE_BYTES))].into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{Kind, assemble, leak_probe, tests::cfg};

    fn drafts() -> Vec<Draft> {
        generate(&cfg(Kind::Career)).unwrap().drafts
    }

    fn of<'a>(ds: &'a [Draft], family: &str) -> Vec<&'a Draft> {
        ds.iter().filter(|d| d.family_id == family).collect()
    }

    fn gold(d: &Draft) -> &str {
        match d.gold {
            Gold::Option(i) => &d.options[i],
            Gold::Noul => panic!("a career row abstains"),
        }
    }

    // -- the fixture ------------------------------------------------------------------------

    #[test]
    fn the_fixture_is_the_contract_by_digest() {
        assert_eq!(sha256_hex(FIXTURE_BYTES), FIXTURE_SHA256);
        assert_eq!(
            FIXTURE_SHA256,
            "0cf04ad791d67fb04bf4e1c5be9b3206f61cf356dddf0a3054c9a8b5a13be47d"
        );
    }

    #[test]
    fn the_fixture_holds_the_four_families_with_their_exact_strings() {
        let specs = fixture().unwrap();
        let got: Vec<(&str, &str, &str, Vec<&str>)> = specs
            .iter()
            .map(|s| {
                (
                    s.task.as_str(),
                    s.question.as_str(),
                    s.slot.as_str(),
                    s.options.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (
                    "career.jd_role_fit",
                    "How well does this job posting fit the target roles stated before the title? If none are stated, compare it with early-career software, ML or systems engineering roles.",
                    "role_fit",
                    vec![
                        "strong_fit",
                        "partial_fit",
                        "adjacent",
                        "mismatch_domain",
                        "mismatch_seniority"
                    ],
                ),
                (
                    "career.jd_sponsorship",
                    "What does this job posting say about visa sponsorship and work authorization?",
                    "sponsorship",
                    vec![
                        "sponsors_visa",
                        "no_sponsorship",
                        "citizenship_or_clearance_required",
                        "not_stated"
                    ],
                ),
                (
                    "career.jd_work_mode",
                    "Where does this job posting say the work is done?",
                    "work_mode",
                    vec!["remote", "hybrid", "onsite", "not_stated"],
                ),
                (
                    "career.jd_seniority",
                    "What level of seniority does this job posting ask for?",
                    "seniority",
                    vec![
                        "intern_or_new_grad",
                        "mid",
                        "senior",
                        "staff_plus",
                        "manager"
                    ],
                ),
            ]
        );
    }

    #[test]
    fn the_generators_options_are_the_fixtures_and_every_table_class_is_one() {
        let specs = fixture().unwrap();
        for (family, table) in [
            (SPONSORSHIP_FAMILY, SPONSORSHIP),
            (WORK_MODE_FAMILY, WORK_MODE),
            (SENIORITY_FAMILY, SENIORITY),
        ] {
            let spec = specs.iter().find(|s| s.task == family).unwrap();
            let classes: Vec<&str> = table.iter().map(|(c, _)| *c).collect();
            let options: Vec<&str> = spec.options.iter().map(String::as_str).collect();
            assert_eq!(classes, options, "{family}");
        }
    }

    #[test]
    fn the_bare_context_list_names_exactly_these_families() {
        for f in FAMILIES {
            assert!(
                decisions::BARE_CONTEXT_FAMILIES.contains(&f),
                "{f} is not a bare-context family"
            );
        }
    }

    // -- Vani's Render, hand-written cases ---------------------------------------------------

    #[test]
    fn render_without_a_target_is_the_title_location_and_description() {
        assert_eq!(
            render("Backend Engineer", "Austin, TX", "Build services.", ""),
            (
                "Title: Backend Engineer\nLocation: Austin, TX\n\nBuild services.".to_owned(),
                false
            )
        );
    }

    #[test]
    fn render_with_a_target_puts_the_header_first() {
        // Vani's TestTriagerRendersTargetSummaryIntoContext golden.
        assert_eq!(
            render("x", "y", "z", "ML systems / inference roles"),
            (
                "Target roles: ML systems / inference roles\n\nTitle: x\nLocation: y\n\nz"
                    .to_owned(),
                false
            )
        );
        assert_eq!(
            render("T", "", "d\n\nmore", "a b"),
            (
                "Target roles: a b\n\nTitle: T\nLocation: \n\nd\n\nmore".to_owned(),
                false
            )
        );
    }

    #[test]
    fn render_cuts_at_the_cap_back_to_a_rune_boundary() {
        let overhead = "Title: T\nLocation: \n\n".len();
        // Exactly the cap: not cut.
        let exact = "a".repeat(MAX_CONTEXT_BYTES - overhead);
        let (got, cut) = render("T", "", &exact, "");
        assert_eq!((got.len(), cut), (MAX_CONTEXT_BYTES, false));
        // One byte over: cut to the cap.
        let over = "a".repeat(MAX_CONTEXT_BYTES - overhead + 1);
        let (got, cut) = render("T", "", &over, "");
        assert_eq!((got.len(), cut), (MAX_CONTEXT_BYTES, true));
        // A 2-, 3- and 4-byte character straddling the cap: the cut backs up to before it.
        for ch in ['é', '€', '😀'] {
            let width = ch.len_utf8();
            for straddle in 1..width {
                // `straddle` bytes of the character sit before the cap, the rest after it.
                let lead = MAX_CONTEXT_BYTES - overhead - straddle;
                let description = format!("{}{ch}tail", "a".repeat(lead));
                let (got, cut) = render("T", "", &description, "");
                assert!(cut, "{ch:?}/{straddle}");
                assert_eq!(got.len(), MAX_CONTEXT_BYTES - straddle, "{ch:?}/{straddle}");
                assert!(got.ends_with('a'), "{ch:?}/{straddle}");
                assert!(!got.contains('\u{fffd}'));
            }
            // A whole character ending exactly at the cap is kept.
            let lead = MAX_CONTEXT_BYTES - overhead - width;
            let description = format!("{}{ch}tail", "a".repeat(lead));
            let (got, cut) = render("T", "", &description, "");
            assert!(cut && got.ends_with(ch), "{ch:?}");
            assert_eq!(got.len(), MAX_CONTEXT_BYTES);
        }
        // The header counts toward the cap.
        let (got, cut) = render("T", "", &"é".repeat(MAX_CONTEXT_BYTES), "a b");
        assert!(cut && got.len() <= MAX_CONTEXT_BYTES && got.starts_with("Target roles: a b"));
    }

    #[test]
    fn a_summary_is_normalized_as_vani_does() {
        assert_eq!(normalize_summary("  a \t b\n\nc  ").unwrap(), "a b c");
        assert_eq!(normalize_summary("   ").unwrap(), "");
        assert!(normalize_summary(&"x".repeat(MAX_SUMMARY_BYTES)).is_ok());
        assert!(normalize_summary(&"x".repeat(MAX_SUMMARY_BYTES + 1)).is_err());
    }

    // -- the rows ---------------------------------------------------------------------------

    #[test]
    fn every_gold_is_one_of_its_familys_options_and_the_shape_is_bare() {
        let specs = fixture().unwrap();
        let ds = drafts();
        assert_eq!(
            ds.iter()
                .map(|d| d.family_id)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        for d in &ds {
            let spec = specs.iter().find(|s| s.task == d.family_id).unwrap();
            assert_eq!(d.options, spec.options, "{}", d.template);
            assert_eq!(d.slot_name, spec.slot);
            assert!(spec.options.iter().any(|o| o == gold(d)), "{}", d.template);
            assert_eq!(d.stratum, format!("{}/{}", d.family_id, gold(d)));
            assert_eq!(d.template_class, gold(d));
            assert!(
                d.question.is_empty(),
                "a bare-context family has no question"
            );
            assert!(d.fixed_options);
            assert!(d.context.len() <= MAX_CONTEXT_BYTES);
            assert!(
                !d.context.contains('{') && !d.context.contains('}'),
                "{}",
                d.context
            );
            assert!(!d.context.contains('@') && !d.context.contains("://"));
            assert!(d.context.contains("Title: ") && d.context.contains("\nLocation: "));
        }
    }

    #[test]
    fn every_option_has_at_least_four_templates_in_the_tables_and_in_the_drafts() {
        for (family, table) in [
            (SPONSORSHIP_FAMILY, SPONSORSHIP),
            (WORK_MODE_FAMILY, WORK_MODE),
            (SENIORITY_FAMILY, SENIORITY),
        ] {
            for (class, templates) in table {
                assert!(
                    templates.len() >= 4,
                    "{family}/{class}: {}",
                    templates.len()
                );
            }
        }
        assert!(LAYOUTS.len() >= 4);
        let ds = drafts();
        for spec in fixture().unwrap() {
            for option in &spec.options {
                let templates: std::collections::BTreeSet<&str> = ds
                    .iter()
                    .filter(|d| d.family_id == spec.task && gold(d) == option)
                    .map(|d| d.template.as_str())
                    .collect();
                assert!(
                    templates.len() >= 4,
                    "{}/{option}: {} template(s)",
                    spec.task,
                    templates.len()
                );
            }
        }
    }

    #[test]
    fn splits_hold_whole_templates_and_every_option_has_val_and_train_templates() {
        let c = cfg(Kind::Career);
        let a = assemble(&c, &generate(&c).unwrap().drafts).unwrap();
        let mut side: BTreeMap<(&str, &str), std::collections::BTreeSet<&str>> = BTreeMap::new();
        for r in &a.candidates {
            side.entry((r.family_id.as_str(), r.group_key.as_str()))
                .or_default()
                .insert(r.split);
        }
        assert!(
            side.values().all(|s| s.len() == 1),
            "a template straddles the split"
        );
        for spec in fixture().unwrap() {
            for option in &spec.options {
                let mut val = std::collections::BTreeSet::new();
                let mut train = std::collections::BTreeSet::new();
                for r in a.candidates.iter().filter(|r| {
                    r.family_id == spec.task
                        && r.options[match r.gold {
                            Gold::Option(i) => i,
                            Gold::Noul => unreachable!("no career row abstains"),
                        }] == *option
                }) {
                    match r.split {
                        "val" => val.insert(r.group_key.as_str()),
                        _ => train.insert(r.group_key.as_str()),
                    };
                }
                assert!(!val.is_empty(), "{}/{option}: no val template", spec.task);
                assert!(
                    train.len() >= 3,
                    "{}/{option}: {} train templates",
                    spec.task,
                    train.len()
                );
                assert!(val.is_disjoint(&train));
            }
        }
        assert!(
            a.report["refused"]
                .get("noul_on_a_family_stating_zero")
                .is_none()
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_rows_and_another_seed_does_not() {
        let key = |ds: &[Draft]| -> Vec<(String, String, String, String)> {
            ds.iter()
                .map(|d| {
                    (
                        d.family_id.to_owned(),
                        d.template.clone(),
                        d.context.clone(),
                        gold(d).to_owned(),
                    )
                })
                .collect()
        };
        let a = key(&drafts());
        assert_eq!(a, key(&drafts()));
        let mut other = cfg(Kind::Career);
        other.seed += 1;
        assert_ne!(a, key(&generate(&other).unwrap().drafts));
    }

    // -- the target header ------------------------------------------------------------------

    fn header(d: &Draft) -> Option<&str> {
        d.context
            .strip_prefix("Target roles: ")
            .map(|rest| rest.split("\n\n").next().unwrap())
    }

    #[test]
    fn the_header_is_in_both_labels_contexts_for_the_three_non_role_fit_families() {
        // Enough rows per template that a fair coin sits well inside the band.
        let mut c = cfg(Kind::Career);
        c.rows_per_template = 40;
        let ds = generate(&c).unwrap().drafts;
        for family in [SPONSORSHIP_FAMILY, WORK_MODE_FAMILY, SENIORITY_FAMILY] {
            let rows = of(&ds, family);
            let mut by_option: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
            for d in &rows {
                let e = by_option.entry(gold(d)).or_default();
                if header(d).is_some() {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
            }
            for (option, (with, without)) in &by_option {
                let share = *with as f64 / (with + without) as f64;
                assert!(
                    *with > 0 && *without > 0 && (0.3..=0.7).contains(&share),
                    "{family}/{option}: {with} with a header, {without} without"
                );
            }
            // Per template too: a template is not a header-bearing one.
            let mut by_template: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
            for d in &rows {
                let e = by_template.entry(d.template.as_str()).or_default();
                if header(d).is_some() {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
            }
            assert!(
                by_template.values().all(|(w, wo)| *w > 0 && *wo > 0),
                "{family}: a template always or never carries a header"
            );
        }
    }

    #[test]
    fn every_header_is_a_normalized_profile_summary() {
        let allowed: std::collections::BTreeSet<String> = all_summaries()
            .iter()
            .map(|s| normalize_summary(s).unwrap())
            .collect();
        for d in drafts() {
            if let Some(h) = header(&d) {
                assert!(allowed.contains(h), "{h:?}");
                assert_eq!(normalize_summary(h).unwrap(), h);
            }
        }
        // The messy summary in the data is what exercises the normalization.
        assert!(
            all_summaries()
                .iter()
                .any(|s| normalize_summary(s).unwrap() != *s)
        );
    }

    // -- role fit ---------------------------------------------------------------------------

    #[test]
    fn every_profile_meets_every_role_fit_label() {
        let specs = fixture().unwrap();
        let spec = specs.iter().find(|s| s.task == ROLE_FIT_FAMILY).unwrap();
        for p in PROFILES {
            for label in &spec.options {
                assert!(
                    ROLES
                        .iter()
                        .any(|r| role_fit_label(p, r) == Some(label.as_str())),
                    "profile {} has no {label} role",
                    p.id
                );
            }
        }
    }

    #[test]
    fn one_role_takes_different_labels_under_different_targets() {
        let by_id = |id: &str| PROFILES.iter().find(|p| p.id == id).unwrap();
        let by_title = |t: &str| ROLES.iter().find(|r| r.title == t).unwrap();
        let ml = by_title("Machine Learning Engineer, Inference");
        assert_eq!(role_fit_label(by_id("default"), ml), Some("strong_fit"));
        assert_eq!(role_fit_label(by_id("ml_systems"), ml), Some("strong_fit"));
        assert_eq!(role_fit_label(by_id("embedded"), ml), Some("adjacent"));
        assert_eq!(
            role_fit_label(by_id("security"), ml),
            Some("mismatch_domain")
        );
        let nurse = by_title("Registered Nurse, Medical-Surgical Unit");
        assert!(
            PROFILES
                .iter()
                .all(|p| role_fit_label(p, nurse) == Some("mismatch_domain"))
        );
        let split = by_title("Applied ML Engineer and Sales Engineer");
        assert_eq!(
            role_fit_label(by_id("ml_systems"), split),
            Some("partial_fit")
        );
        assert_eq!(role_fit_label(by_id("security"), split), None);
        let principal = by_title("Principal Backend Engineer");
        assert_eq!(
            role_fit_label(by_id("default"), principal),
            Some("mismatch_seniority")
        );
        assert_eq!(role_fit_label(by_id("ml_systems"), principal), None);
    }

    #[test]
    fn a_senior_title_over_graduate_duties_and_the_reverse_are_in_the_catalogue() {
        let grad = ROLES
            .iter()
            .find(|r| r.title.starts_with("Senior") && r.level == Entry)
            .expect("a Senior title at an early-career level");
        assert!(grad.reqs.contains("new graduates"));
        let assoc = ROLES
            .iter()
            .find(|r| r.title.starts_with("Associate") && r.level == Far)
            .expect("an Associate title at a far-above level");
        assert!(assoc.reqs.contains("10+ years"));
        for id in [
            "senior-title-grad",
            "associate-8yrs",
            "senior-manager",
            "team-lead-reports",
        ] {
            assert!(
                SENIORITY
                    .iter()
                    .flat_map(|(_, ts)| ts.iter())
                    .any(|t| t.id == id),
                "{id}"
            );
        }
    }

    /// The gold re-derived from the rendered context alone: the header names the profile (an
    /// absent header is `default`), the title line names the role, and the rule gives the label.
    #[test]
    fn a_role_fit_gold_is_rederived_from_the_context_and_the_empty_summary_is_the_default() {
        let ds = drafts();
        let (mut with, mut without) = (0usize, 0usize);
        for d in of(&ds, ROLE_FIT_FAMILY) {
            let profile = match header(d) {
                None => {
                    without += 1;
                    &PROFILES[0]
                }
                Some(h) => {
                    with += 1;
                    PROFILES
                        .iter()
                        .find(|p| {
                            p.summaries
                                .iter()
                                .any(|s| normalize_summary(s).unwrap() == h)
                        })
                        .unwrap_or_else(|| panic!("{h:?} is no profile's summary"))
                }
            };
            let title_line = d
                .context
                .split("Title: ")
                .nth(1)
                .unwrap()
                .split('\n')
                .next()
                .unwrap();
            let roles: Vec<&Role> = ROLES
                .iter()
                .filter(|r| {
                    title_line == r.title
                        || title_line.starts_with(&format!("{}, ", r.title))
                        || title_line.starts_with(&format!("{} - ", r.title))
                })
                .collect();
            assert_eq!(roles.len(), 1, "{title_line}");
            assert_eq!(
                role_fit_label(profile, roles[0]),
                Some(gold(d)),
                "{}",
                d.context
            );
        }
        assert!(with > 0 && without > 0);
        let share = without as f64 / (with + without) as f64;
        assert!((0.08..=0.3).contains(&share), "empty-summary share {share}");
        // Every label has rows both with and without a header.
        for label in [
            "strong_fit",
            "partial_fit",
            "adjacent",
            "mismatch_domain",
            "mismatch_seniority",
        ] {
            let rows: Vec<&Draft> = of(&ds, ROLE_FIT_FAMILY)
                .into_iter()
                .filter(|d| gold(d) == label)
                .collect();
            assert!(rows.iter().any(|d| header(d).is_some()), "{label}");
            assert!(rows.iter().any(|d| header(d).is_none()), "{label}");
        }
    }

    // -- hard cases -------------------------------------------------------------------------

    fn text_of(table: Table, class: &str) -> Vec<String> {
        table
            .iter()
            .filter(|(c, _)| *c == class)
            .flat_map(|(_, ts)| ts.iter())
            .map(|t| format!("{} {}", t.location, t.body).to_lowercase())
            .collect()
    }

    #[test]
    fn not_stated_postings_mention_neighbours_and_never_the_topic() {
        for text in text_of(SPONSORSHIP, "not_stated") {
            let text = text.replace("citizenship status", "");
            for word in [
                "visa",
                "work permit",
                "authoriz",
                "authoris",
                "clearance",
                "immigration",
                "right to work",
                "sponsorship",
                "eligible to work",
                "citizen",
                "work for any employer",
            ] {
                assert!(!text.contains(word), "{word:?} in {text}");
            }
        }
        for text in text_of(WORK_MODE, "not_stated") {
            for word in [
                "remote",
                "hybrid",
                "on-site",
                "onsite",
                "from home",
                "work from",
                "in person",
                "in-person",
                "days a week",
                "every day",
                "in the office",
                "home-office",
                "commut",
            ] {
                assert!(!text.contains(word), "{word:?} in {text}");
            }
        }
        let all = |table, class| text_of(table, class).join("\n");
        // The neighbouring topics the brief names, present.
        assert!(all(SPONSORSHIP, "not_stated").contains("relocation stipend"));
        assert!(all(SPONSORSHIP, "not_stated").contains("sponsors regional hackathons"));
        assert!(all(WORK_MODE, "not_stated").contains("our office is at"));
        assert!(all(WORK_MODE, "not_stated").contains("headquartered in"));
    }

    #[test]
    fn sponsorship_named_only_to_deny_it_and_the_remote_versus_hybrid_cases_exist() {
        let no = text_of(SPONSORSHIP, "no_sponsorship").join("\n");
        assert!(
            no.contains("sponsors local hackathons")
                && no.contains("not eligible for visa sponsorship")
        );
        assert!(no.contains("visa sponsorship is not provided"));
        let remote = text_of(WORK_MODE, "remote").join("\n");
        assert!(remote.contains("remote within the united states"));
        assert!(remote.contains("offsite"));
        let hybrid = text_of(WORK_MODE, "hybrid").join("\n");
        assert!(
            hybrid.contains("remote on monday and friday") && hybrid.contains("remote-friendly")
        );
        for text in text_of(SPONSORSHIP, "sponsors_visa") {
            assert!(!text.contains("unable to sponsor") && !text.contains("not eligible"));
        }
    }

    // -- privacy and the leak probe ---------------------------------------------------------

    #[test]
    fn nothing_names_a_person_or_a_profile_value() {
        for d in drafts() {
            let lower = d.context.to_lowercase();
            assert!(!lower.contains("bharath") && !lower.contains("vani"));
        }
    }

    /// The break-it-first pair's second half (the first is
    /// `synth::tests::the_probe_reads_a_planted_label_leak`): each family, scored on held-out
    /// templates, stays under the bound its config records. Career is gated like email and
    /// Jarvis: no ruling grants it report-only.
    #[test]
    fn each_family_stays_under_the_leak_bound() {
        let c = cfg(Kind::Career);
        let a = assemble(&c, &generate(&c).unwrap().drafts).unwrap();
        for family in FAMILIES {
            let rows: Vec<&crate::decisions::Candidate> = a
                .candidates
                .iter()
                .filter(|r| r.family_id == family)
                .collect();
            let p = leak_probe(&c, family, &rows, 2).unwrap();
            assert_eq!(p["state"], "ran", "{family}");
            assert_eq!(p["passed"], true, "{family}: {p}");
        }
    }

    /// The whole pool path: decontamination, the gated probes and the manifest. Every probe is
    /// `passed: true` (none is report-only), the request's context is the rendered posting, and
    /// the options stay in the fixture's order.
    #[test]
    fn a_career_pool_builds_with_gated_probes_and_bare_context_rows() {
        let c = cfg(Kind::Career);
        let g = generate(&c).unwrap();
        let target: crate::decisions::TargetSet = (
            "stand-in".to_owned(),
            vec![(
                "t0".to_owned(),
                "lighthouse keepers counting migrating cranes over a frozen northern estuary at dawn"
                    .to_owned(),
            )],
        );
        let built = crate::synth::build(
            &c,
            "sha",
            &g.drafts,
            &[target],
            g.inputs.clone(),
            &g.report,
            1,
        )
        .unwrap();
        for family in FAMILIES {
            assert_eq!(
                built.manifest["leak_probe"][family]["passed"], true,
                "{family}"
            );
            assert!(built.manifest["leak_probe"][family]["report_only"].is_null());
        }
        assert_eq!(built.manifest["source_id"], "lappi/synth-career");
        assert_eq!(built.manifest["inputs"]["career_fixture"], FIXTURE_SHA256);
        assert_eq!(
            built.manifest["generator"]["fixture_sha256"],
            FIXTURE_SHA256
        );
        let specs = fixture().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for line in built
            .examples
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
        {
            let row: Value = serde_json::from_slice(line).unwrap();
            let family = row["family_id"].as_str().unwrap().to_owned();
            let spec = specs.iter().find(|s| s.task == family).unwrap();
            let ctx = row["context"].as_str().unwrap();
            assert!(
                ctx.starts_with("Title: ") || ctx.starts_with("Target roles: "),
                "{family}: a question was rendered in front: {ctx}"
            );
            assert_eq!(row["slot_name"], spec.slot.as_str());
            let options: Vec<&str> = row["options"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap())
                .collect();
            assert_eq!(
                options, spec.options,
                "{family}: options are in the caller's order"
            );
            seen.insert(family);
        }
        assert_eq!(seen.len(), 4);
    }
}
