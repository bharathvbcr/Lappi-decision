//! Jarvis's decisions, synthesised by rule (see [`crate::synth`]). Three families:
//!
//! - **`jarvis.target`**: which on-screen element performs a step. The options are the
//!   enumerated elements of a synthetic accessibility tree.
//!   - The tree holds the control under one of several labels. This is the drift Jarvis's
//!     overlays exist for: "Search" vs "Find member".
//!   - It also holds same-word decoys ("Search history", "Advanced search"), controls for
//!     other steps, and text fields. A text field named like a button, `text_field "Search"`,
//!     is in the decoy set.
//!   - The gold is `noul` when the tree lacks the control (Jarvis's `missing-control`
//!     scenario).
//! - **`jarvis.field_fill`**: which saved-profile entry fills a form field, or "leave this
//!   field empty".
//!   - Only profile *keys* are named; no profile value is ever written, real or synthetic.
//!   - Secrets, payment fields and someone else's details (an emergency contact) are left
//!     empty by rule.
//!   - A label that does not say what it wants is `noul`.
//! - **`jarvis.step_risk`**: what a policy requires before a proposed step runs. The policy is
//!   modelled on `policies/jarvis-bank.json`: editable fields, sensitive targets, permitted
//!   step kinds, unattended change.
//!   - Rule 7: the options *classify* what the policy requires. None of them is an
//!     authorisation, and Manvi's policy check stays the authority.
//!   - A step whose target is not on screen is `noul`.
//!
//! **Not read:** Jarvis's scenario inputs and contracts. Nothing here generates a member id, a
//! balance or a currency amount, so the scenarios' literal values (`M-1001`, `M-1002`, `M-9999`,
//! `125000`) cannot occur. A test checks this, and `--target` scans the contracts as well. The
//! bank's two tenant layouts are 2 of 10 screens, so the families cannot memorise them.

use std::collections::BTreeSet;

use crate::decisions::Gold;
use crate::synth::{Config, Draft, Scope};

pub const TARGET: &str = "jarvis.target";
pub const FIELD_FILL: &str = "jarvis.field_fill";
pub const STEP_RISK: &str = "jarvis.step_risk";

/// The share of `jarvis.target` rows whose tree lacks the control (gold `noul`).
const TARGET_ABSENT_RATE: f64 = 0.12;
/// The share of `jarvis.step_risk` rows whose target is not on screen (gold `noul`).
const RISK_ABSENT_RATE: f64 = 0.08;

const WINDOWS: [&str; 10] = [
    "Jarvis Bank \u{2014} North Cooperative",
    "Jarvis Bank \u{2014} South Mutual",
    "Acme Payroll \u{2014} Employees",
    "Riverside Clinic \u{2014} Patient intake",
    "Contoso Travel \u{2014} Booking",
    "System Settings \u{2014} Accounts",
    "Shipping details",
    "Create account",
    "Member services",
    "Expense report",
];

/// A control: its role, the labels it appears under, same-word decoys, and how a step asks
/// for it (each phrasing is one template).
struct Concept {
    id: &'static str,
    role: &'static str,
    labels: &'static [&'static str],
    decoys: &'static [&'static str],
    intents: &'static [&'static str],
}

const CONCEPTS: [Concept; 12] = [
    Concept {
        id: "lookup",
        role: "button",
        labels: &["Search", "Find member", "Look up", "Find", "Look up member"],
        decoys: &[
            "Search history",
            "Advanced search",
            "Clear search",
            "Saved searches",
        ],
        intents: &[
            "Look up the member whose ID was just entered.",
            "Run the search for the entered member ID.",
            "Find the account that matches the member ID in the field.",
            "Start the member lookup.",
            "Search for the record using the ID typed above.",
        ],
    },
    Concept {
        id: "create_subaccount",
        role: "button",
        labels: &[
            "New subaccount",
            "Add subaccount",
            "Create subaccount",
            "Open a subaccount",
        ],
        decoys: &[
            "Subaccount settings",
            "Delete subaccount",
            "Subaccount list",
        ],
        intents: &[
            "Start creating a new subaccount.",
            "Open the form for adding a subaccount.",
            "Begin setting up another subaccount for this member.",
            "Add a subaccount to this member's profile.",
            "Create a fresh subaccount.",
        ],
    },
    Concept {
        id: "review",
        role: "button",
        labels: &["Review", "Review details", "Preview", "Check details"],
        decoys: &["Reviews", "Review policy", "Write a review"],
        intents: &[
            "Open the summary of the entered details before anything is submitted.",
            "Go to the review screen.",
            "Check the entered details on the review page.",
            "Preview what will be submitted.",
            "Show the details for a final look before confirming.",
        ],
    },
    Concept {
        id: "back",
        role: "button",
        labels: &["Back", "Previous", "Go back"],
        decoys: &["Back to top", "Feedback", "Backup"],
        intents: &[
            "Return to the previous screen.",
            "Go back one step.",
            "Leave this screen for the one before it.",
            "Navigate back without changing anything.",
            "Step back to the earlier page.",
        ],
    },
    Concept {
        id: "dismiss",
        role: "button",
        labels: &["Dismiss", "Close", "Not now", "Got it"],
        decoys: &["Dismiss all notifications", "Close account", "Closed items"],
        intents: &[
            "Close the notice that is covering the form.",
            "Get rid of the pop-up banner.",
            "Dismiss the overlay message.",
            "Hide the notification that appeared.",
            "Clear the alert from the screen.",
        ],
    },
    Concept {
        id: "continue",
        role: "button",
        labels: &["Continue", "Next", "Proceed"],
        decoys: &["Continue later", "Continue shopping", "Next month"],
        intents: &[
            "Move on to the next step of the flow.",
            "Proceed past this page.",
            "Go to the next page of the wizard.",
            "Advance to the following step.",
            "Carry on with the process.",
        ],
    },
    Concept {
        id: "submit",
        role: "button",
        labels: &["Submit", "Send", "Send request"],
        decoys: &["Submit feedback", "Send a copy to me", "Submitted items"],
        intents: &[
            "Send the completed form.",
            "Submit the entered information.",
            "Finalise and send the request.",
            "Send the form off for processing.",
            "Hand in the filled form.",
        ],
    },
    Concept {
        id: "save",
        role: "button",
        labels: &["Save", "Save changes", "Apply"],
        decoys: &["Save as draft", "Saved items", "Apply filter"],
        intents: &[
            "Store the changes made on this page.",
            "Keep the edits that were just made.",
            "Save the updated settings.",
            "Persist the current changes.",
            "Commit the modified values on this page.",
        ],
    },
    Concept {
        id: "cancel",
        role: "button",
        labels: &["Cancel", "Discard"],
        decoys: &[
            "Cancel subscription",
            "Cancellation policy",
            "Discard all drafts",
        ],
        intents: &[
            "Abandon this form without saving.",
            "Stop the current action and throw it away.",
            "Back out of this dialog, discarding the edits.",
            "Cancel what is in progress.",
            "Drop the entry without keeping it.",
        ],
    },
    Concept {
        id: "sign_in",
        role: "button",
        labels: &["Sign in", "Log in"],
        decoys: &["Sign-in help", "Login history", "Sign up"],
        intents: &[
            "Log into the account with the entered credentials.",
            "Sign the user in.",
            "Submit the login details.",
            "Start a signed-in session.",
            "Authenticate with the details typed above.",
        ],
    },
    Concept {
        id: "export",
        role: "button",
        labels: &["Export", "Download CSV", "Export table"],
        decoys: &["Export settings", "Import", "Download app"],
        intents: &[
            "Download the table as a file.",
            "Export the current list.",
            "Save the results to a spreadsheet file.",
            "Get a copy of this data as a download.",
            "Pull the records out to a file.",
        ],
    },
    Concept {
        id: "help",
        role: "link",
        labels: &["Help", "Support", "Help center"],
        decoys: &["Helpful links", "Help us improve", "Supported devices"],
        intents: &[
            "Open the help resources.",
            "Find the support page.",
            "Get assistance for this screen.",
            "Show the documentation for this page.",
            "Reach the help center.",
        ],
    },
];

/// Concepts too close to tell apart from a one-line step; a tree for one never carries the
/// other as filler, so the gold is decidable.
const CONFLICTS: [(&str, &str); 6] = [
    ("dismiss", "cancel"),
    ("back", "cancel"),
    ("submit", "continue"),
    ("save", "submit"),
    ("review", "continue"),
    ("sign_in", "submit"),
];

/// Text fields a tree may carry. "Search" is here on purpose: `text_field "Search"` beside
/// `button "Search"`.
const TREE_FIELDS: [&str; 8] = [
    "Member ID",
    "Subaccount name",
    "Email",
    "Amount",
    "Notes",
    "Date",
    "Reference",
    "Search",
];

fn conflicts(a: &str, b: &str) -> bool {
    CONFLICTS
        .iter()
        .any(|(x, y)| (*x == a && *y == b) || (*x == b && *y == a))
}

fn element(role: &str, name: &str) -> String {
    format!("{role} \"{name}\"")
}

/// `jarvis.target`: one template per (control, phrasing).
fn target_drafts(cfg: &Config, out: &mut Vec<Draft>) -> Result<(), String> {
    for (ci, c) in CONCEPTS.iter().enumerate() {
        for (pi, intent) in c.intents.iter().enumerate() {
            let template = format!("target:{}:{pi}", c.id);
            for r in 0..cfg.rows_per_template {
                let s = Scope::new(cfg.seed, &format!("jarvis\u{1f}{template}\u{1f}{r}"));
                let absent = s.unit("absent") < TARGET_ABSENT_RATE;
                let mut names: BTreeSet<String> = BTreeSet::new();
                let mut elements: Vec<String> = Vec::new();
                let mut push = |role: &str, name: &str, names: &mut BTreeSet<String>| {
                    let e = element(role, name);
                    if names.insert(e.clone()) {
                        elements.push(e);
                    }
                };
                let gold_element = element(c.role, s.pick("label", c.labels));
                if !absent {
                    push(c.role, s.pick("label", c.labels), &mut names);
                }
                for name in s
                    .shuffle("decoys", c.decoys)
                    .iter()
                    .take(1 + s.below("n_decoys", 2))
                {
                    push("button", name, &mut names);
                }
                let fillers: Vec<&Concept> = CONCEPTS
                    .iter()
                    .enumerate()
                    .filter(|(j, o)| *j != ci && !conflicts(c.id, o.id))
                    .map(|(_, o)| o)
                    .collect();
                for o in s
                    .shuffle("fillers", &fillers)
                    .iter()
                    .take(3 + s.below("n_fill", 5))
                {
                    push(
                        o.role,
                        s.pick(&format!("fill-label-{}", o.id), o.labels),
                        &mut names,
                    );
                }
                for f in s
                    .shuffle("fields", &TREE_FIELDS)
                    .iter()
                    .take(1 + s.below("n_fields", 3))
                {
                    push("text_field", f, &mut names);
                }
                let gold = if absent {
                    Gold::Noul
                } else {
                    Gold::Option(
                        elements
                            .iter()
                            .position(|e| *e == gold_element)
                            .ok_or("gold element missing")?,
                    )
                };
                let focused = s.pick("focused", &TREE_FIELDS);
                let context = format!(
                    "Window: {}\nFocused: text_field \"{focused}\"\nStep: {intent}",
                    s.pick("window", &WINDOWS)
                );
                out.push(Draft {
                    family_id: TARGET,
                    stratum: format!("{TARGET}/{}", if absent { "absent" } else { "present" }),
                    template: template.clone(),
                    template_class: c.id.to_owned(),
                    context,
                    question: "Which on-screen element performs this step?".to_owned(),
                    slot_name: "element",
                    options: elements,
                    gold,
                    fixed_options: false,
                });
            }
        }
    }
    Ok(())
}

/// `jarvis.field_fill`'s options: the saved profile's keys, and the refusal to fill.
pub const PROFILE_OPTIONS: [&str; 14] = [
    "given name",
    "family name",
    "full name",
    "email address",
    "phone number",
    "street address",
    "address line 2",
    "city",
    "state or region",
    "postal code",
    "country",
    "date of birth",
    "organization",
    "leave this field empty",
];
const LEAVE_EMPTY: usize = 13;
const NAME_CLASSES: [usize; 3] = [0, 1, 2];

/// Per profile key: the labels a form may give it. Each label is one template.
const FIELD_LABELS: [&[&str]; 13] = [
    &[
        "First name",
        "Given name",
        "First",
        "Forename",
        "Your first name",
    ],
    &[
        "Last name",
        "Surname",
        "Family name",
        "Last",
        "Your last name",
    ],
    &[
        "Full name",
        "Name",
        "Your name",
        "Name as it appears on your ID",
        "Full legal name",
    ],
    &[
        "Email",
        "Email address",
        "E-mail",
        "Your email",
        "Contact email",
    ],
    &[
        "Phone",
        "Phone number",
        "Mobile number",
        "Telephone",
        "Cell phone",
    ],
    &[
        "Street address",
        "Address line 1",
        "Address",
        "Street",
        "House number and street",
    ],
    &[
        "Address line 2",
        "Apartment, suite, etc.",
        "Apt / Unit",
        "Suite or floor",
        "Flat number",
    ],
    &["City", "Town or city", "Town", "Locality", "City / Town"],
    &["State", "Province", "State / Province", "Region", "County"],
    &[
        "ZIP code",
        "Postal code",
        "Postcode",
        "ZIP",
        "ZIP / Postal code",
    ],
    &[
        "Country",
        "Country or region",
        "Nation",
        "Country of residence",
        "Select country",
    ],
    &[
        "Date of birth",
        "Birthday",
        "DOB",
        "Birth date",
        "When were you born?",
    ],
    &[
        "Company",
        "Organization",
        "Employer",
        "Company name",
        "Organisation",
    ],
];

/// Fields left empty whatever the profile holds: `(template id, section, label)`. Someone
/// else's details, payment data, secrets and one-time codes are never autofilled.
const LEAVE_EMPTY_FIELDS: [(&str, &str, &str); 8] = [
    ("emergency-phone", "Emergency contact", "Phone"),
    ("emergency-name", "Emergency contact", "Full name"),
    ("card-name", "Payment", "Name on card"),
    ("card-number", "Payment", "Card number"),
    ("passcode", "Sign-in", "Passcode"),
    ("otp", "Verification", "Verification code"),
    ("promo", "Order summary", "Promo code"),
    ("security", "Security question", "Mother's maiden name"),
];

/// Labels that do not say what they want.
const AMBIGUOUS_LABELS: [&str; 6] = [
    "Number",
    "Details",
    "Other",
    "Info",
    "Value",
    "Name or company",
];

const FORMS: [&str; 8] = [
    "Checkout",
    "Create your account",
    "Job application",
    "Membership sign-up",
    "Event registration",
    "Delivery details",
    "Patient intake",
    "Subaccount setup",
];
const NEUTRAL_SECTIONS: [&str; 5] = [
    "Your details",
    "Contact information",
    "Shipping address",
    "Billing address",
    "Account",
];

fn field_context(s: &Scope, section: &str, label: &str, gold_class: Option<usize>) -> String {
    // Sibling labels never come from the name classes, so "Name" beside "Last name" (which would
    // make "Name" a given name) cannot occur, and never from the gold's own class.
    let siblings: Vec<&str> = FIELD_LABELS
        .iter()
        .enumerate()
        .filter(|(i, _)| !NAME_CLASSES.contains(i) && Some(*i) != gold_class)
        .map(|(i, labels)| *s.pick(&format!("sibling-label-{i}"), labels))
        .filter(|l| *l != label)
        .collect();
    let chosen = s.shuffle("siblings", &siblings);
    let n = 1 + s.below("n_siblings", 3);
    let sib: Vec<String> = chosen.iter().take(n).map(|l| format!("\"{l}\"")).collect();
    format!(
        "Form: {}\nSection: {section}\nField: text_field \"{label}\"\nOther fields in this section: {}",
        s.pick("form", &FORMS),
        sib.join(", ")
    )
}

fn field_draft(template: String, class: &str, context: String, gold: Gold) -> Draft {
    Draft {
        family_id: FIELD_FILL,
        stratum: format!(
            "{FIELD_FILL}/{}",
            match gold {
                Gold::Noul => "ambiguous",
                Gold::Option(LEAVE_EMPTY) => "leave_empty",
                Gold::Option(_) => "fill",
            }
        ),
        template,
        template_class: class.to_owned(),
        context,
        question: "Which entry from the user's saved profile should fill this field?".to_owned(),
        slot_name: "profile_entry",
        options: PROFILE_OPTIONS.iter().map(|o| (*o).to_owned()).collect(),
        gold,
        fixed_options: true,
    }
}

fn field_drafts(cfg: &Config, out: &mut Vec<Draft>) -> Result<(), String> {
    for (k, labels) in FIELD_LABELS.iter().enumerate() {
        for (li, label) in labels.iter().enumerate() {
            let template = format!("field:{}:{li}", PROFILE_OPTIONS[k].replace(' ', "_"));
            for r in 0..cfg.rows_per_template {
                let s = Scope::new(cfg.seed, &format!("jarvis\u{1f}{template}\u{1f}{r}"));
                let ctx = field_context(&s, s.pick("section", &NEUTRAL_SECTIONS), label, Some(k));
                out.push(field_draft(
                    template.clone(),
                    PROFILE_OPTIONS[k],
                    ctx,
                    Gold::Option(k),
                ));
            }
        }
    }
    for (id, section, label) in LEAVE_EMPTY_FIELDS {
        let template = format!("field:leave_empty:{id}");
        for r in 0..cfg.rows_per_template {
            let s = Scope::new(cfg.seed, &format!("jarvis\u{1f}{template}\u{1f}{r}"));
            let ctx = field_context(&s, section, label, None);
            out.push(field_draft(
                template.clone(),
                PROFILE_OPTIONS[LEAVE_EMPTY],
                ctx,
                Gold::Option(LEAVE_EMPTY),
            ));
        }
    }
    for (ai, label) in AMBIGUOUS_LABELS.iter().enumerate() {
        let template = format!("field:noul:{ai}");
        for r in 0..cfg.rows_per_template {
            let s = Scope::new(cfg.seed, &format!("jarvis\u{1f}{template}\u{1f}{r}"));
            let ctx = field_context(&s, s.pick("section", &NEUTRAL_SECTIONS), label, None);
            out.push(field_draft(template.clone(), "noul", ctx, Gold::Noul));
        }
    }
    Ok(())
}

/// `jarvis.step_risk`'s options: what the policy requires, never a grant (rule 7).
pub const RISK_OPTIONS: [&str; 4] = [
    "Within the policy as a read-only step.",
    "Within the policy, but a human must confirm this change before it runs.",
    "Within the policy as an unattended change.",
    "Outside the policy: this step kind or target is not permitted.",
];
const RISK_CLASSES: [&str; 5] = [
    "read_only",
    "confirm_change",
    "unattended_change",
    "outside",
    "noul",
];
const READ_KINDS: [&str; 4] = ["press", "click", "extract", "assert"];
const CHANGE_KINDS: [&str; 2] = ["set_value", "type_text"];
const ALL_KINDS: [&str; 10] = [
    "press",
    "click",
    "set_value",
    "type_text",
    "scroll",
    "extract",
    "assert",
    "branch",
    "wait",
    "conclude",
];
const RISK_FIELDS: [&str; 10] = [
    "Member ID",
    "Subaccount name",
    "Passcode",
    "Email",
    "Amount",
    "Notes",
    "Reference",
    "Phone",
    "Payee",
    "Memo",
];
const RISK_BUTTONS: [&str; 8] = [
    "Search",
    "Find member",
    "Review",
    "Back",
    "Dismiss",
    "Continue",
    "Export",
    "Help",
];

/// A synthetic policy and the screen it governs.
struct Policy {
    fields_on_screen: Vec<&'static str>,
    buttons_on_screen: Vec<&'static str>,
    editable: Vec<&'static str>,
    sensitive: Vec<&'static str>,
    read_only_targets: Vec<String>,
    permitted: Vec<&'static str>,
    unattended_change: bool,
}

/// One proposed step: its kind, its target's role and name, and whether the target is on screen.
struct Step {
    kind: &'static str,
    role: &'static str,
    name: &'static str,
}

/// The class a step falls in under a policy: the single rule every row's gold comes from.
fn classify(p: &Policy, st: &Step) -> &'static str {
    let on_screen = match st.role {
        "text_field" => p.fields_on_screen.contains(&st.name),
        _ => p.buttons_on_screen.contains(&st.name),
    };
    if !on_screen {
        return "noul";
    }
    if !p.permitted.contains(&st.kind) {
        return "outside";
    }
    if CHANGE_KINDS.contains(&st.kind) {
        if st.role != "text_field" || !p.editable.contains(&st.name) {
            return "outside";
        }
        if p.sensitive.contains(&st.name) || !p.unattended_change {
            return "confirm_change";
        }
        return "unattended_change";
    }
    let listed = p.read_only_targets.contains(&element(st.role, st.name))
        || (st.role == "text_field" && p.editable.contains(&st.name));
    if listed { "read_only" } else { "outside" }
}

fn draw_policy(s: &Scope) -> Policy {
    let fields: Vec<&'static str> = s
        .shuffle("fields", &RISK_FIELDS)
        .into_iter()
        .take(3 + s.below("n_fields", 4))
        .collect();
    let buttons: Vec<&'static str> = s
        .shuffle("buttons", &RISK_BUTTONS)
        .into_iter()
        .take(3 + s.below("n_buttons", 4))
        .collect();
    let editable: Vec<&'static str> = fields
        .iter()
        .copied()
        .filter(|f| s.unit(&format!("editable-{f}")) < 0.6)
        .collect();
    let sensitive: Vec<&'static str> = editable
        .iter()
        .copied()
        .filter(|f| *f == "Passcode" || s.unit(&format!("sensitive-{f}")) < 0.35)
        .collect();
    let mut read_only_targets: Vec<String> = buttons
        .iter()
        .filter(|b| s.unit(&format!("listed-{b}")) < 0.85)
        .map(|b| element("button", b))
        .collect();
    read_only_targets.extend(
        fields
            .iter()
            .filter(|f| s.unit(&format!("listed-{f}")) < 0.7)
            .map(|f| element("text_field", f)),
    );
    let permitted: Vec<&'static str> = ALL_KINDS
        .iter()
        .copied()
        .filter(|k| s.unit(&format!("kind-{k}")) < 0.75)
        .collect();
    Policy {
        fields_on_screen: fields,
        buttons_on_screen: buttons,
        editable,
        sensitive,
        read_only_targets,
        permitted,
        unattended_change: s.unit("unattended") < 0.4,
    }
}

fn draw_step(s: &Scope, p: &Policy, absent: bool) -> Step {
    let change = s.unit("change") < 0.5;
    let kind = if s.unit("odd_kind") < 0.15 {
        *s.pick("kind_any", &ALL_KINDS)
    } else if change {
        *s.pick("kind_change", &CHANGE_KINDS)
    } else {
        *s.pick("kind_read", &READ_KINDS)
    };
    let field = CHANGE_KINDS.contains(&kind) || s.unit("field_target") < 0.4;
    let (role, pool, all): (&'static str, &[&'static str], &[&'static str]) = if field {
        (
            "text_field",
            p.fields_on_screen.as_slice(),
            &RISK_FIELDS[..],
        )
    } else {
        ("button", p.buttons_on_screen.as_slice(), &RISK_BUTTONS[..])
    };
    let name = if absent {
        let off: Vec<&'static str> = all.iter().copied().filter(|n| !pool.contains(n)).collect();
        if off.is_empty() {
            *s.pick("name", pool)
        } else {
            *s.pick("name_off", &off)
        }
    } else {
        *s.pick("name", pool)
    };
    Step { kind, role, name }
}

fn list(xs: &[String]) -> String {
    if xs.is_empty() {
        "none".to_owned()
    } else {
        xs.join(", ")
    }
}

fn quoted(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|x| format!("\"{x}\"")).collect()
}

/// The five ways a policy and step are rendered; each is a template per class.
fn render(fmt: usize, p: &Policy, st: &Step) -> String {
    let screen: Vec<String> = p
        .buttons_on_screen
        .iter()
        .map(|b| element("button", b))
        .chain(p.fields_on_screen.iter().map(|f| element("text_field", f)))
        .collect();
    let step = element(st.role, st.name);
    let unattended = if p.unattended_change {
        "allowed"
    } else {
        "not allowed"
    };
    match fmt {
        0 => format!(
            "Screen: {}\nPolicy: editable fields {}; sensitive targets {}; read-only targets {}; permitted step kinds {}; unattended change {unattended}.\nProposed step: {} on {step}",
            list(&screen),
            list(&quoted(&p.editable)),
            list(&quoted(&p.sensitive)),
            list(&p.read_only_targets),
            p.permitted.join(", "),
            st.kind
        ),
        1 => format!(
            "The window shows {}. The policy lets these fields be edited: {}; it marks {} as sensitive, lists {} as targets that may be read, permits the step kinds {}, and unattended changes are {unattended}. Jarvis proposes to {} {step}.",
            list(&screen),
            list(&quoted(&p.editable)),
            list(&quoted(&p.sensitive)),
            list(&p.read_only_targets),
            p.permitted.join(", "),
            st.kind
        ),
        2 => format!(
            "- on screen: {}\n- editable_fields: {}\n- sensitive_targets: {}\n- read_only_targets: {}\n- permitted_step_kinds: {}\n- unattended_change: {}\n- step: {{\"kind\": \"{}\", \"target\": {step}}}",
            list(&screen),
            list(&quoted(&p.editable)),
            list(&quoted(&p.sensitive)),
            list(&p.read_only_targets),
            p.permitted.join(", "),
            p.unattended_change,
            st.kind
        ),
        3 => format!(
            "STEP {} -> {step}\nVISIBLE {}\nEDITABLE {}\nSENSITIVE {}\nREADABLE {}\nKINDS {}\nUNATTENDED {}",
            st.kind,
            list(&screen),
            list(&quoted(&p.editable)),
            list(&quoted(&p.sensitive)),
            list(&p.read_only_targets),
            p.permitted.join(" "),
            if p.unattended_change { "yes" } else { "no" }
        ),
        _ => format!(
            "Proposed: {} {step}. Visible elements: {}. Policy says editable = [{}], sensitive = [{}], readable = [{}], kinds = [{}], unattended changes {unattended}.",
            st.kind,
            list(&screen),
            p.editable.join(", "),
            p.sensitive.join(", "),
            list(&p.read_only_targets),
            p.permitted.join(", ")
        ),
    }
}

/// Attempts to land one row in a class before the template is reported short.
const RISK_ATTEMPTS: usize = 1_000;

fn risk_drafts(cfg: &Config, out: &mut Vec<Draft>) -> Result<(), String> {
    for class in RISK_CLASSES {
        for fmt in 0..5 {
            let template = format!("risk:{class}:{fmt}");
            for r in 0..cfg.rows_per_template {
                // Draw policy and step until they fall in this template's class, so every class
                // is filled whatever its natural rate; the gold is still classify()'s, never set.
                let mut made = None;
                for a in 0..RISK_ATTEMPTS {
                    let s = Scope::new(
                        cfg.seed,
                        &format!("jarvis\u{1f}{template}\u{1f}{r}\u{1f}{a}"),
                    );
                    let p = draw_policy(&s);
                    let absent = class == "noul" || s.unit("absent") < RISK_ABSENT_RATE;
                    let st = draw_step(&s, &p, absent);
                    if classify(&p, &st) == class {
                        made = Some(render(fmt, &p, &st));
                        break;
                    }
                }
                let context = made.ok_or_else(|| {
                    format!(
                        "{template}: no draw landed in class {class} in {RISK_ATTEMPTS} attempts"
                    )
                })?;
                let gold = match RISK_CLASSES.iter().position(|c| *c == class) {
                    Some(4) => Gold::Noul,
                    Some(i) => Gold::Option(i),
                    None => return Err(format!("unknown risk class {class}")),
                };
                out.push(Draft {
                    family_id: STEP_RISK,
                    stratum: format!("{STEP_RISK}/{class}"),
                    template: template.clone(),
                    template_class: class.to_owned(),
                    context,
                    question: "What does the policy require before this step runs?".to_owned(),
                    slot_name: "policy_requirement",
                    options: RISK_OPTIONS.iter().map(|o| (*o).to_owned()).collect(),
                    gold,
                    fixed_options: true,
                });
            }
        }
    }
    Ok(())
}

pub fn drafts(cfg: &Config) -> Result<Vec<Draft>, String> {
    let mut out = Vec::new();
    target_drafts(cfg, &mut out)?;
    field_drafts(cfg, &mut out)?;
    risk_drafts(cfg, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::synth::{Kind, assemble, leak_probe, tests::cfg};

    #[test]
    fn every_class_of_every_family_has_four_templates() {
        let ds = drafts(&cfg(Kind::Jarvis)).unwrap();
        let mut by: BTreeMap<(&str, &str), BTreeSet<&str>> = BTreeMap::new();
        for d in &ds {
            by.entry((d.family_id, &d.template_class))
                .or_default()
                .insert(&d.template);
        }
        for (k, ts) in &by {
            assert!(ts.len() >= 4, "{k:?}: {} templates", ts.len());
        }
        assemble(&cfg(Kind::Jarvis), &ds).unwrap();
    }

    #[test]
    fn the_scenarios_literal_values_never_occur() {
        let ds = drafts(&cfg(Kind::Jarvis)).unwrap();
        for d in &ds {
            for lit in ["M-1001", "M-1002", "M-9999", "125000", "USD"] {
                assert!(!d.context.contains(lit), "{lit} in {}", d.context);
            }
        }
    }

    /// Rule 7: no option grants anything.
    #[test]
    fn no_option_is_an_authorisation() {
        for o in RISK_OPTIONS.iter().chain(PROFILE_OPTIONS.iter()) {
            let l = o.to_lowercase();
            for w in [
                "allow", "approve", "execute", "authoris", "authoriz", "pass",
            ] {
                assert!(!l.contains(w), "{o:?} contains {w:?}");
            }
        }
    }

    #[test]
    fn a_target_rows_gold_is_its_control_or_noul_when_absent() {
        let ds = drafts(&cfg(Kind::Jarvis)).unwrap();
        let mut absent = 0;
        for d in ds.iter().filter(|d| d.family_id == TARGET) {
            let c = CONCEPTS.iter().find(|c| c.id == d.template_class).unwrap();
            match d.gold {
                Gold::Option(i) => {
                    assert!(c.labels.iter().any(|l| d.options[i] == element(c.role, l)))
                }
                Gold::Noul => {
                    absent += 1;
                    assert!(
                        !d.options
                            .iter()
                            .any(|o| c.labels.iter().any(|l| *o == element(c.role, l)))
                    );
                }
            }
        }
        assert!(absent > 0);
    }

    #[test]
    fn the_risk_rule_on_named_cases() {
        let p = Policy {
            fields_on_screen: vec!["Member ID", "Passcode", "Notes"],
            buttons_on_screen: vec!["Search"],
            editable: vec!["Member ID", "Passcode"],
            sensitive: vec!["Passcode"],
            read_only_targets: vec![element("button", "Search")],
            permitted: vec!["click", "set_value"],
            unattended_change: true,
        };
        let st = |kind, role, name| Step { kind, role, name };
        assert_eq!(classify(&p, &st("click", "button", "Search")), "read_only");
        assert_eq!(
            classify(&p, &st("set_value", "text_field", "Member ID")),
            "unattended_change"
        );
        assert_eq!(
            classify(&p, &st("set_value", "text_field", "Passcode")),
            "confirm_change"
        );
        assert_eq!(
            classify(&p, &st("set_value", "text_field", "Notes")),
            "outside"
        );
        assert_eq!(
            classify(&p, &st("type_text", "text_field", "Member ID")),
            "outside"
        );
        assert_eq!(classify(&p, &st("click", "button", "Export")), "noul");
        let q = Policy {
            unattended_change: false,
            ..p
        };
        assert_eq!(
            classify(&q, &st("set_value", "text_field", "Member ID")),
            "confirm_change"
        );
    }

    #[test]
    fn the_shipped_fixed_option_families_stay_under_the_leak_bound() {
        let c = cfg(Kind::Jarvis);
        let a = assemble(&c, &drafts(&c).unwrap()).unwrap();
        for family in [FIELD_FILL, STEP_RISK] {
            let rows: Vec<&crate::decisions::Candidate> = a
                .candidates
                .iter()
                .filter(|r| r.family_id == family)
                .collect();
            let p = leak_probe(&c, family, &rows, 2).unwrap();
            assert_eq!(p["state"], "ran", "{family}: {p}");
            assert_eq!(p["passed"], true, "{family}: {p}");
        }
    }
}
