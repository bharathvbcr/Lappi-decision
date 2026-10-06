//! Tool selection for the human's own agents and apps, synthesised by rule (see
//! [`crate::synth`]).
//!
//! **Inputs.** Two pinned data files, read rather than embedded:
//! - `data/synth/tool-catalog-*.json`: each app's tool catalog, extracted mechanically from the
//!   app's canonical source. The owner's handle is replaced, and each tool has an authored
//!   confusable `group`.
//! - `data/synth/tool-phrasings-*.json`: per tool name, complete and underspecified requests,
//!   plus general questions no tool serves.
//!
//! **Row shape.** Exactly lane 4's `when2call.tool_select` / `toolace.tool_select` shape
//! (`convert_tools.rs`), so converted and synthetic rows are one task:
//! - the context is `Available tools:` with one JSON tool object per line, then `User: <request>`;
//! - the options are the offered tool names, then the three actions [`ASK`], [`UNABLE`],
//!   [`DIRECT`];
//! - the slot is `action`.
//!
//! **Labels by construction** (Fable's ruling, 2026-10-06):
//! - **call:** a complete phrasing, with its tool offered.
//! - **ask:** an underspecified phrasing (a required argument is absent), with its tool offered.
//! - **unable:** a complete phrasing with the tool's whole confusable group removed from the
//!   offer. Removing only the tool would leave a near-substitute, and "unable" would be wrong.
//! - **direct:** a general question; the offer is drawn normally.
//! - **`noul`:** a stated zero for this family. Ambiguity goes to "ask".
//!
//! **Offer.** One app's catalog, never mixed across catalogs:
//! - at most [`MAX_OFFERED`] tools, so that tools plus the three actions fit
//!   `decisions::MAX_OPTIONS`;
//! - the gold always, with at least one same-group sibling when it has one.
//!
//! A caller with a larger catalog pre-filters (`docs/caller-contract.md`).
//!
//! **Masking.** A seeded share of rows replaces tool names with `tool_1..tool_k` in both the
//! tool list and the options, so the model must read descriptions. The caller sends real names,
//! so unmasked rows stay the majority.
//!
//! **Leak control.** The options differ per row, so the class-label probe does not apply.
//! Instead, every complete phrasing's content-word overlap with its gold tool's name tokens and
//! description is measured, its distribution is reported, and a file whose median exceeds the
//! config's bound is refused.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::decisions::{self, Gold};
use crate::sha256::sha256_hex;
use crate::synth::{Config, Draft, Generated, Scope};

pub const FAMILY: &str = "apps.tool_select";
pub const CATALOG_SCHEMA: &str = "qd-tool-catalog/v1";
pub const PHRASINGS_SCHEMA: &str = "qd-tool-phrasings/v1";
/// Lane 4's three actions and question, verbatim (`convert_tools.rs` ASK, UNABLE, DIRECT,
/// QUESTION). Converge on one owner when the lanes merge.
pub const ASK: &str = "Ask the user a clarifying question";
pub const UNABLE: &str = "Say it cannot help with the available tools";
pub const DIRECT: &str = "Answer directly without calling a tool";
pub const QUESTION: &str = "What should the assistant do next: call one of the listed tools (by \
     name), ask the user a clarifying question, say it cannot help with the available tools, or \
     answer directly?";
/// Tools offered per row: plus the three actions, `decisions::MAX_OPTIONS` (16).
pub const MAX_OFFERED: usize = decisions::MAX_OPTIONS - 3;
const MIN_OFFERED: usize = 4;
/// Bounds on what the data files may hold.
const MAX_FILE_BYTES: u64 = 16 << 20;
const MAX_DESCRIPTION_BYTES: usize = 600;
const STOPWORDS: [&str; 48] = [
    "the", "and", "for", "with", "that", "this", "from", "what", "which", "who", "how", "are",
    "was", "were", "has", "have", "had", "can", "you", "your", "its", "into", "onto", "about",
    "all", "any", "our", "out", "not", "but", "does", "did", "will", "would", "should", "could",
    "there", "their", "them", "then", "than", "when", "where", "why", "just", "get", "give", "me",
];

/// The tool-selection generator's inputs and bounds (`"tools"` in a `qd-synth-config/v1`).
#[derive(Clone, Debug)]
pub struct ToolsConfig {
    pub catalog: PathBuf,
    pub catalog_sha256: String,
    pub phrasings: PathBuf,
    pub phrasings_sha256: String,
    /// The share of complete-phrasing rows turned into "unable" rows.
    pub unable_rate: f64,
    /// The share of rows whose tool names are masked.
    pub mask_rate: f64,
    /// The bound on the median content-word overlap of complete phrasings with their tool.
    pub max_median_overlap: f64,
    /// Tools never offered and never gold, each with its reason: a tool that *is* one of the
    /// three actions (a tool that asks the user a question makes "ask" and "call" both right).
    pub exclude_tools: BTreeMap<String, String>,
}

impl ToolsConfig {
    pub fn parse(v: &Value) -> Result<Self, String> {
        use decisions::{get_f64, get_str};
        let what = "config.tools";
        let unit = |name: &str| -> Result<f64, String> {
            let x = get_f64(v, name, what)?;
            if (0.0..=1.0).contains(&x) {
                Ok(x)
            } else {
                Err(format!("{what}.{name} {x} is outside [0, 1]"))
            }
        };
        Ok(Self {
            catalog: PathBuf::from(get_str(v, "catalog", what)?),
            catalog_sha256: get_str(v, "catalog_sha256", what)?.to_owned(),
            phrasings: PathBuf::from(get_str(v, "phrasings", what)?),
            phrasings_sha256: get_str(v, "phrasings_sha256", what)?.to_owned(),
            unable_rate: unit("unable_rate")?,
            mask_rate: unit("mask_rate")?,
            max_median_overlap: unit("max_median_overlap")?,
            exclude_tools: match v.get("exclude_tools") {
                None => BTreeMap::new(),
                Some(x) => x
                    .as_object()
                    .ok_or_else(|| format!("{what}.exclude_tools is not an object"))?
                    .iter()
                    .map(|(k, r)| match r.as_str() {
                        Some(r) if !r.trim().is_empty() => Ok((k.clone(), r.to_owned())),
                        _ => Err(format!(
                            "{what}.exclude_tools.{k}: the reason is not a non-empty string"
                        )),
                    })
                    .collect::<Result<_, String>>()?,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: String,
    pub required: bool,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub params: Vec<Param>,
    pub group: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Catalog {
    pub app: String,
    pub tools: Vec<Tool>,
}

#[derive(Clone, Debug, Default)]
pub struct ToolPhrasings {
    pub complete: Vec<String>,
    /// `(text, the required parameter it leaves out)`.
    pub underspecified: Vec<(String, String)>,
}

#[derive(Clone, Debug)]
pub struct Phrasings {
    pub by_tool: BTreeMap<String, ToolPhrasings>,
    pub direct: Vec<String>,
}

fn read_pinned(path: &Path, want: &str) -> Result<(Value, String), String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if size > MAX_FILE_BYTES {
        return Err(format!(
            "{}: {size} bytes; the bound is {MAX_FILE_BYTES}",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = sha256_hex(&bytes);
    decisions::check_pin(path, &got, want, "the config")?;
    let v = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((v, got))
}

fn str_of<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a str, String> {
    decisions::get_str(v, key, what)
}

fn list_of<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a Vec<Value>, String> {
    decisions::get(v, key, what)?
        .as_array()
        .ok_or_else(|| format!("{what}: {key:?} is not a list"))
}

/// The catalogs, checked: unique app names, unique tool names within an app, bounded
/// descriptions, every non-null group with at least two members in its catalog.
pub fn parse_catalogs(v: &Value) -> Result<Vec<Catalog>, String> {
    if str_of(v, "schema", "catalog")? != CATALOG_SCHEMA {
        return Err(format!("catalog: schema is not {CATALOG_SCHEMA:?}"));
    }
    let mut apps = BTreeSet::new();
    let mut out = Vec::new();
    for c in list_of(v, "catalogs", "catalog")? {
        let app = str_of(c, "app", "catalog entry")?.to_owned();
        if !apps.insert(app.clone()) {
            return Err(format!("catalog: app {app:?} appears twice"));
        }
        let what = format!("catalog {app}");
        let mut names = BTreeSet::new();
        let mut tools = Vec::new();
        for t in list_of(c, "tools", &what)? {
            let name = str_of(t, "name", &what)?.to_owned();
            if name.trim().is_empty() || !names.insert(name.clone()) {
                return Err(format!("{what}: tool name {name:?} is empty or repeated"));
            }
            let description = str_of(t, "description", &what)?.to_owned();
            if description.trim().is_empty() || description.len() > MAX_DESCRIPTION_BYTES {
                return Err(format!(
                    "{what}/{name}: description is empty or over {MAX_DESCRIPTION_BYTES} bytes"
                ));
            }
            let params = list_of(t, "params", &what)?
                .iter()
                .map(|p| {
                    Ok(Param {
                        name: str_of(p, "name", &what)?.to_owned(),
                        ty: str_of(p, "type", &what)?.to_owned(),
                        required: p.get("required").and_then(Value::as_bool).ok_or_else(|| {
                            format!("{what}/{name}: a param's \"required\" is not a bool")
                        })?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let group = match t.get("group") {
                None | Some(Value::Null) => None,
                Some(Value::String(g)) if !g.trim().is_empty() => Some(g.clone()),
                Some(_) => {
                    return Err(format!(
                        "{what}/{name}: group is not a non-empty string or null"
                    ));
                }
            };
            tools.push(Tool {
                name,
                description,
                params,
                group,
            });
        }
        if tools.is_empty() {
            return Err(format!("{what}: no tools"));
        }
        let mut sizes: BTreeMap<&str, usize> = BTreeMap::new();
        for t in &tools {
            if let Some(g) = &t.group {
                *sizes.entry(g.as_str()).or_default() += 1;
            }
        }
        if let Some((g, _)) = sizes.iter().find(|(_, n)| **n < 2) {
            return Err(format!(
                "{what}: group {g:?} has one member; a confusable set needs two"
            ));
        }
        out.push(Catalog { app, tools });
    }
    if out.is_empty() {
        return Err("catalog: no catalogs".into());
    }
    Ok(out)
}

/// The phrasings, checked against the catalogs: every catalog tool has an entry with at least
/// four complete phrasings; in each catalog where a tool has a required parameter, at least two
/// of its underspecified phrasings name one that is required there; every underspecified
/// phrasing names a parameter required in at least one catalog; no text appears twice; at
/// least four direct questions.
///
/// Required parameters differ by catalog for a shared name (devcouncil's `renew_lease` takes
/// `task_id`; manvi's takes nothing), so an underspecified phrasing is an "ask" row only in a
/// catalog where its missing parameter is required ([`drafts_from`]).
pub fn parse_phrasings(v: &Value, catalogs: &[Catalog]) -> Result<Phrasings, String> {
    if str_of(v, "schema", "phrasings")? != PHRASINGS_SCHEMA {
        return Err(format!("phrasings: schema is not {PHRASINGS_SCHEMA:?}"));
    }
    let mut seen = BTreeSet::new();
    let mut fresh = |text: &str, what: &str| -> Result<String, String> {
        let t = text.trim();
        if t.is_empty() {
            return Err(format!("{what}: an empty phrasing"));
        }
        if !seen.insert(t.to_lowercase()) {
            return Err(format!("{what}: {t:?} appears twice in the file"));
        }
        Ok(t.to_owned())
    };
    let mut by_tool: BTreeMap<String, ToolPhrasings> = BTreeMap::new();
    for e in list_of(v, "tools", "phrasings")? {
        let tool = str_of(e, "tool", "phrasings entry")?.to_owned();
        let what = format!("phrasings/{tool}");
        let complete = list_of(e, "complete", &what)?
            .iter()
            .map(|t| {
                t.as_str()
                    .ok_or_else(|| format!("{what}: a complete phrasing is not a string"))
                    .and_then(|t| fresh(t, &what))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let underspecified = list_of(e, "underspecified", &what)?
            .iter()
            .map(|u| {
                Ok((
                    fresh(str_of(u, "text", &what)?, &what)?,
                    str_of(u, "missing", &what)?.to_owned(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if by_tool
            .insert(
                tool.clone(),
                ToolPhrasings {
                    complete,
                    underspecified,
                },
            )
            .is_some()
        {
            return Err(format!("{what}: the tool has two entries"));
        }
    }
    let direct = list_of(v, "direct", "phrasings")?
        .iter()
        .map(|t| {
            t.as_str()
                .ok_or_else(|| "phrasings: a direct question is not a string".to_owned())
                .and_then(|t| fresh(t, "phrasings/direct"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if direct.len() < crate::synth::MIN_TEMPLATES_PER_CLASS {
        return Err(format!(
            "phrasings: {} direct questions; at least {} are required",
            direct.len(),
            crate::synth::MIN_TEMPLATES_PER_CLASS
        ));
    }
    let mut required_anywhere: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for c in catalogs {
        for t in &c.tools {
            required_anywhere
                .entry(t.name.as_str())
                .or_default()
                .extend(required_params(t));
        }
    }
    for c in catalogs {
        for t in &c.tools {
            let what = format!("phrasings/{} (catalog {})", t.name, c.app);
            let p = by_tool
                .get(&t.name)
                .ok_or_else(|| format!("{what}: no entry"))?;
            if p.complete.len() < crate::synth::MIN_TEMPLATES_PER_CLASS {
                return Err(format!(
                    "{what}: {} complete phrasings; at least {} are required",
                    p.complete.len(),
                    crate::synth::MIN_TEMPLATES_PER_CLASS
                ));
            }
            // The phrasing-level check first: it names the offending text and parameter, where
            // the count below could only say how many were short.
            for (text, missing) in &p.underspecified {
                if !required_anywhere[t.name.as_str()].contains(missing.as_str()) {
                    return Err(format!(
                        "{what}: {text:?} names {missing:?}, which no catalog requires"
                    ));
                }
            }
            let required = required_params(t);
            let here = p
                .underspecified
                .iter()
                .filter(|(_, m)| required.contains(m.as_str()))
                .count();
            if !required.is_empty() && here < 2 {
                return Err(format!(
                    "{what}: a required parameter and {here} underspecified phrasings naming one; at least 2 are required"
                ));
            }
        }
    }
    let known: BTreeSet<&str> = catalogs
        .iter()
        .flat_map(|c| c.tools.iter().map(|t| t.name.as_str()))
        .collect();
    if let Some(stray) = by_tool.keys().find(|k| !known.contains(k.as_str())) {
        return Err(format!("phrasings/{stray}: no catalog has this tool"));
    }
    Ok(Phrasings { by_tool, direct })
}

fn required_params(t: &Tool) -> BTreeSet<&str> {
    t.params
        .iter()
        .filter(|p| p.required)
        .map(|p| p.name.as_str())
        .collect()
}

/// The catalogs and phrasings without the config's excluded tools. Every excluded name must be
/// a catalog tool: a stale exclusion is refused, not ignored.
pub fn exclude(
    catalogs: Vec<Catalog>,
    mut ph: Phrasings,
    excluded: &BTreeMap<String, String>,
) -> Result<(Vec<Catalog>, Phrasings), String> {
    for name in excluded.keys() {
        if !catalogs
            .iter()
            .any(|c| c.tools.iter().any(|t| &t.name == name))
        {
            return Err(format!("exclude_tools: {name:?} is in no catalog"));
        }
        ph.by_tool.remove(name);
    }
    let mut out = Vec::new();
    for mut c in catalogs {
        c.tools.retain(|t| !excluded.contains_key(&t.name));
        if c.tools.is_empty() {
            return Err(format!("catalog {}: every tool is excluded", c.app));
        }
        out.push(c);
    }
    Ok((out, ph))
}

/// Lowercase alphabetic words of three letters or more, stopwords removed.
fn content_words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| w.len() >= 3)
        .map(str::to_lowercase)
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// A tool name's words: split on `_ . - /` and at lower-to-upper camel-case boundaries.
fn name_tokens(name: &str) -> Vec<String> {
    let mut spaced = String::with_capacity(name.len() + 8);
    let mut prev_lower = false;
    for ch in name.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            spaced.push(' ');
        }
        prev_lower = ch.is_ascii_lowercase();
        spaced.push(if matches!(ch, '_' | '.' | '-' | '/') {
            ' '
        } else {
            ch
        });
    }
    content_words(&spaced)
}

/// The share of `phrasing`'s content words that appear in `tool`'s name tokens or description.
/// A phrasing with no content words overlaps by 0.
pub fn overlap(phrasing: &str, tool: &Tool) -> f64 {
    let words = content_words(phrasing);
    if words.is_empty() {
        return 0.0;
    }
    let vocab: BTreeSet<String> = name_tokens(&tool.name)
        .into_iter()
        .chain(content_words(&tool.description))
        .collect();
    words.iter().filter(|w| vocab.contains(w.as_str())).count() as f64 / words.len() as f64
}

/// The overlap distribution over every (catalog tool, complete phrasing) pair, and whether its
/// median is within `bound`.
pub fn overlap_report(catalogs: &[Catalog], ph: &Phrasings, bound: f64) -> (Value, bool) {
    let mut xs: Vec<f64> = Vec::new();
    for c in catalogs {
        for t in &c.tools {
            if let Some(p) = ph.by_tool.get(&t.name) {
                xs.extend(p.complete.iter().map(|s| overlap(s, t)));
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    let q = |f: f64| {
        xs.get(((xs.len().max(1) - 1) as f64 * f).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let median = q(0.5);
    let ok = median <= bound;
    (
        json!({"pairs": xs.len(), "median": median, "p90": q(0.9), "max": q(1.0), "bound": bound, "passed": ok}),
        ok,
    )
}

fn tool_json(t: &Tool, shown_name: &str) -> String {
    let props: serde_json::Map<String, Value> = t
        .params
        .iter()
        .map(|p| (p.name.clone(), json!({"type": p.ty})))
        .collect();
    let required: Vec<&str> = t
        .params
        .iter()
        .filter(|p| p.required)
        .map(|p| p.name.as_str())
        .collect();
    json!({
        "name": shown_name,
        "description": t.description,
        "parameters": {"type": "object", "properties": props, "required": required},
    })
    .to_string()
}

/// The tools offered with `gold` (an index into `cat.tools`), in listing order: the gold, at
/// least one same-group sibling when it has one, and fillers from the rest of the catalog.
fn offer(s: &Scope, cat: &Catalog, gold: Option<usize>) -> Vec<usize> {
    let n_cat = cat.tools.len();
    let want = (MIN_OFFERED + s.below("n_offer", MAX_OFFERED - MIN_OFFERED + 1)).min(n_cat);
    let mut chosen: Vec<usize> = Vec::new();
    if let Some(g) = gold {
        chosen.push(g);
        if let Some(group) = &cat.tools[g].group {
            let sibs: Vec<usize> = (0..n_cat)
                .filter(|&i| i != g && cat.tools[i].group.as_ref() == Some(group))
                .collect();
            if !sibs.is_empty() {
                let take = 1 + s.below("n_sibs", sibs.len().min(2));
                chosen.extend(s.shuffle("sibs", &sibs).into_iter().take(take));
            }
        }
    }
    let rest: Vec<usize> = (0..n_cat).filter(|i| !chosen.contains(i)).collect();
    let room = want.saturating_sub(chosen.len());
    chosen.extend(s.shuffle("fill", &rest).into_iter().take(room));
    s.shuffle("order", &chosen)
}

/// One row: the offered tools (indices into `cat.tools`), the request, and the gold, which is a
/// tool index or one of the three actions.
enum Gold3 {
    Tool(usize),
    Ask,
    Unable,
    Direct,
}

#[allow(clippy::too_many_arguments)]
fn row(
    s: &Scope,
    cat: &Catalog,
    offered: &[usize],
    request: &str,
    gold: Gold3,
    template: &str,
    class: &str,
    kind: &str,
    mask_rate: f64,
) -> Result<Draft, String> {
    let masked = s.unit("mask") < mask_rate;
    let shown: Vec<String> = offered
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            if masked {
                format!("tool_{}", k + 1)
            } else {
                cat.tools[i].name.clone()
            }
        })
        .collect();
    let mut context = String::from("Available tools:\n");
    if offered.is_empty() {
        context.push_str("(none)\n");
    }
    for (k, &i) in offered.iter().enumerate() {
        context.push_str(&tool_json(&cat.tools[i], &shown[k]));
        context.push('\n');
    }
    context.push_str("\nUser: ");
    context.push_str(request);
    let mut options = shown.clone();
    options.extend([ASK, UNABLE, DIRECT].map(str::to_owned));
    let n = offered.len();
    let gold = Gold::Option(match gold {
        Gold3::Tool(i) => offered
            .iter()
            .position(|&j| j == i)
            .ok_or("the gold tool is not offered")?,
        Gold3::Ask => n,
        Gold3::Unable => n + 1,
        Gold3::Direct => n + 2,
    });
    Ok(Draft {
        family_id: FAMILY,
        stratum: format!(
            "{FAMILY}/{}/{kind}{}",
            cat.app,
            if masked { "/masked" } else { "" }
        ),
        template: template.to_owned(),
        template_class: class.to_owned(),
        context,
        question: QUESTION.to_owned(),
        slot_name: "action",
        options,
        gold,
        fixed_options: false,
    })
}

/// Every draft from the parsed inputs, and the counts the manifest reports.
pub fn drafts_from(
    cfg: &Config,
    tc: &ToolsConfig,
    catalogs: &[Catalog],
    ph: &Phrasings,
) -> Result<(Vec<Draft>, Value), String> {
    let mut out = Vec::new();
    let mut unable_skipped = 0usize;
    // Underspecified phrasings whose missing parameter is not required in this catalog: there
    // the request is complete, so it makes no "ask" row.
    let mut ask_not_required_here = 0usize;
    // The class an underspecified phrasing is stratified under: its tool's first catalog, so a
    // phrasing shared by two catalogs sits on one side of the split.
    let mut first_app: BTreeMap<&str, &str> = BTreeMap::new();
    for c in catalogs {
        for t in &c.tools {
            first_app.entry(t.name.as_str()).or_insert(c.app.as_str());
        }
    }
    for cat in catalogs {
        for (ti, t) in cat.tools.iter().enumerate() {
            let p = &ph.by_tool[&t.name];
            for (pi, text) in p.complete.iter().enumerate() {
                let template = format!("call:{}:{pi}", t.name);
                for r in 0..cfg.rows_per_template {
                    let s = Scope::new(
                        cfg.seed,
                        &format!("tools\u{1f}{}\u{1f}{template}\u{1f}{r}", cat.app),
                    );
                    if s.unit("unable") < tc.unable_rate {
                        // Remove the gold's whole confusable group, then offer from what is left.
                        let banned: Vec<usize> = (0..cat.tools.len())
                            .filter(|&i| {
                                i == ti || (t.group.is_some() && cat.tools[i].group == t.group)
                            })
                            .collect();
                        let left: Vec<usize> = (0..cat.tools.len())
                            .filter(|i| !banned.contains(i))
                            .collect();
                        if left.is_empty() {
                            unable_skipped += 1;
                            continue;
                        }
                        let want = (MIN_OFFERED
                            + s.below("n_offer", MAX_OFFERED - MIN_OFFERED + 1))
                        .min(left.len());
                        let offered: Vec<usize> = s
                            .shuffle("unable-offer", &left)
                            .into_iter()
                            .take(want)
                            .collect();
                        out.push(row(
                            &s,
                            cat,
                            &offered,
                            text,
                            Gold3::Unable,
                            &template,
                            &t.name,
                            "unable",
                            tc.mask_rate,
                        )?);
                    } else {
                        let offered = offer(&s, cat, Some(ti));
                        out.push(row(
                            &s,
                            cat,
                            &offered,
                            text,
                            Gold3::Tool(ti),
                            &template,
                            &t.name,
                            "call",
                            tc.mask_rate,
                        )?);
                    }
                }
            }
            let required = required_params(t);
            for (ui, (text, missing)) in p.underspecified.iter().enumerate() {
                if !required.contains(missing.as_str()) {
                    ask_not_required_here += 1;
                    continue;
                }
                let template = format!("ask:{}:{ui}", t.name);
                let class = format!("ask:{}", first_app[t.name.as_str()]);
                for r in 0..cfg.rows_per_template {
                    let s = Scope::new(
                        cfg.seed,
                        &format!("tools\u{1f}{}\u{1f}{template}\u{1f}{r}", cat.app),
                    );
                    let offered = offer(&s, cat, Some(ti));
                    out.push(row(
                        &s,
                        cat,
                        &offered,
                        text,
                        Gold3::Ask,
                        &template,
                        &class,
                        "ask",
                        tc.mask_rate,
                    )?);
                }
            }
        }
    }
    for (di, text) in ph.direct.iter().enumerate() {
        let template = format!("direct:{di}");
        for r in 0..cfg.rows_per_template {
            let s = Scope::new(cfg.seed, &format!("tools\u{1f}{template}\u{1f}{r}"));
            let cat = s.pick("catalog", catalogs);
            let offered = offer(&s, cat, None);
            out.push(row(
                &s,
                cat,
                &offered,
                text,
                Gold3::Direct,
                &template,
                "direct",
                "direct",
                tc.mask_rate,
            )?);
        }
    }
    let mut offered_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for d in &out {
        *offered_hist.entry(d.options.len() - 3).or_default() += 1;
    }
    let report = json!({
        "catalogs": catalogs.iter().map(|c| (c.app.clone(), json!(c.tools.len()))).collect::<serde_json::Map<_, _>>(),
        "offered_tools_histogram": offered_hist,
        "unable_rows_skipped_no_tool_left": unable_skipped,
        "ask_phrasings_skipped_not_required_in_catalog": ask_not_required_here,
        "noul": "a stated zero for this family: ambiguity is labelled ask",
        "held_out_eval": "none real beyond DevType's ~20 palette query -> command assertions (a target, never trained); verdict.cases.json labels policy verdicts, not routing",
    });
    Ok((out, report))
}

/// `qd-prep synth` for `kind: tools`: read and check the pinned inputs, refuse a phrasing file
/// over its overlap bound, then draft.
pub fn generate(cfg: &Config) -> Result<Generated, String> {
    let tc = cfg
        .tools
        .as_ref()
        .ok_or("kind tools without a tools config")?;
    let (cv, csha) = read_pinned(&tc.catalog, &tc.catalog_sha256)?;
    let (pv, psha) = read_pinned(&tc.phrasings, &tc.phrasings_sha256)?;
    let catalogs = parse_catalogs(&cv)?;
    let ph = parse_phrasings(&pv, &catalogs)?;
    let (catalogs, ph) = exclude(catalogs, ph, &tc.exclude_tools)?;
    let (overlap, ok) = overlap_report(&catalogs, &ph, tc.max_median_overlap);
    if !ok {
        return Err(format!(
            "phrasing overlap over its bound: complete phrasings copy their tool's name or \
             description words, so the gold is readable from surface text. {overlap}"
        ));
    }
    let (drafts, mut report) = drafts_from(cfg, tc, &catalogs, &ph)?;
    report["phrasing_overlap"] = overlap;
    report["excluded_tools"] = json!(tc.exclude_tools);
    let inputs = [
        ("tool_catalog".to_owned(), csha),
        ("tool_phrasings".to_owned(), psha),
    ]
    .into();
    Ok(Generated {
        drafts,
        report,
        inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{Kind, assemble, tests::cfg};

    fn tool(name: &str, desc: &str, required: &[&str], group: Option<&str>) -> Tool {
        Tool {
            name: name.to_owned(),
            description: desc.to_owned(),
            params: required
                .iter()
                .map(|p| Param {
                    name: (*p).to_owned(),
                    ty: "string".to_owned(),
                    required: true,
                })
                .collect(),
            group: group.map(str::to_owned),
        }
    }

    fn catalog() -> Vec<Catalog> {
        vec![Catalog {
            app: "demo".to_owned(),
            tools: vec![
                tool(
                    "graph_impact",
                    "Blast radius of a symbol, walked in reverse.",
                    &["target"],
                    Some("radius"),
                ),
                tool(
                    "graph_dependencies",
                    "Outbound edges of a file or symbol.",
                    &["target"],
                    Some("radius"),
                ),
                tool("graph_search", "Find symbols by name.", &["query"], None),
                tool(
                    "task_next",
                    "The next ready task, without a lease.",
                    &[],
                    None,
                ),
                tool(
                    "repo_status",
                    "Worktrees and changes for a repository.",
                    &[],
                    None,
                ),
            ],
        }]
    }

    fn phrasings(cats: &[Catalog]) -> Phrasings {
        let mut by_tool = BTreeMap::new();
        for t in &cats[0].tools {
            let complete = (0..4)
                .map(|i| format!("request {i} for {}", t.name.replace('_', " x ")))
                .collect();
            let underspecified = t
                .params
                .iter()
                .filter(|p| p.required)
                .flat_map(|p| {
                    (0..2).map(move |i| (format!("vague {i} {}", p.name), p.name.clone()))
                })
                .map(|(s, m)| (format!("{s} {}", t.name.len()), m))
                .collect();
            by_tool.insert(
                t.name.clone(),
                ToolPhrasings {
                    complete,
                    underspecified,
                },
            );
        }
        Phrasings {
            by_tool,
            direct: (0..5)
                .map(|i| format!("general question number {i}?"))
                .collect(),
        }
    }

    fn tc() -> ToolsConfig {
        ToolsConfig {
            catalog: PathBuf::new(),
            catalog_sha256: String::new(),
            phrasings: PathBuf::new(),
            phrasings_sha256: String::new(),
            unable_rate: 0.3,
            mask_rate: 0.3,
            max_median_overlap: 0.35,
            exclude_tools: BTreeMap::new(),
        }
    }

    fn base_cfg() -> Config {
        let mut c = cfg(Kind::Tools);
        c.rows_per_template = 8;
        c
    }

    #[test]
    fn the_options_are_lane_fours_shape() {
        let cats = catalog();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats, &phrasings(&cats)).unwrap();
        for d in &ds {
            let n = d.options.len();
            assert!(n <= decisions::MAX_OPTIONS);
            assert_eq!(&d.options[n - 3..], [ASK, UNABLE, DIRECT]);
            assert!(d.context.starts_with("Available tools:\n") && d.context.contains("\nUser: "));
            assert_eq!(d.slot_name, "action");
        }
    }

    /// An unable row offers no member of the gold's confusable group, so no near-substitute is
    /// left on the list.
    #[test]
    fn an_unable_row_removes_the_whole_group() {
        let cats = catalog();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats, &phrasings(&cats)).unwrap();
        let unable: Vec<&Draft> = ds
            .iter()
            .filter(|d| d.stratum.contains("/unable"))
            .collect();
        assert!(!unable.is_empty());
        for d in unable
            .iter()
            .filter(|d| d.template_class == "graph_impact" && !d.stratum.ends_with("masked"))
        {
            assert!(
                !d.options
                    .iter()
                    .any(|o| o == "graph_impact" || o == "graph_dependencies"),
                "{:?}",
                d.options
            );
            assert_eq!(d.gold, Gold::Option(d.options.len() - 2));
        }
    }

    #[test]
    fn a_call_row_offers_its_gold_and_a_sibling() {
        let cats = catalog();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats, &phrasings(&cats)).unwrap();
        for d in ds
            .iter()
            .filter(|d| d.template_class == "graph_impact" && d.stratum.ends_with("/call"))
        {
            let Gold::Option(g) = d.gold else { panic!() };
            assert_eq!(d.options[g], "graph_impact");
            assert!(d.options.iter().any(|o| o == "graph_dependencies"));
        }
    }

    #[test]
    fn ask_and_direct_rows_point_at_their_actions_and_noul_is_zero() {
        let cats = catalog();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats, &phrasings(&cats)).unwrap();
        assert!(ds.iter().all(|d| d.gold != Gold::Noul));
        for d in &ds {
            let n = d.options.len();
            if d.stratum.contains("/ask") {
                assert_eq!(d.gold, Gold::Option(n - 3));
            }
            if d.stratum.contains("/direct") {
                assert_eq!(d.gold, Gold::Option(n - 1));
            }
        }
    }

    #[test]
    fn a_masked_row_names_no_real_tool() {
        let cats = catalog();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats, &phrasings(&cats)).unwrap();
        let masked: Vec<&Draft> = ds
            .iter()
            .filter(|d| d.stratum.ends_with("/masked"))
            .collect();
        assert!(!masked.is_empty());
        for d in masked {
            for t in &cats[0].tools {
                assert!(!d.context.contains(&format!("\"{}\"", t.name)));
                assert!(!d.options.contains(&t.name));
            }
        }
    }

    #[test]
    fn the_drafts_assemble_under_the_split_rules() {
        let cats = catalog();
        let c = base_cfg();
        let (ds, _) = drafts_from(&c, &tc(), &cats, &phrasings(&cats)).unwrap();
        assemble(&c, &ds).unwrap();
    }

    /// The break-it-first pair: phrasings that copy the description read over the bound and are
    /// refused; paraphrases stay under it.
    #[test]
    fn the_overlap_bound_catches_description_copied_phrasings() {
        let cats = catalog();
        let mut copied = phrasings(&cats);
        for t in &cats[0].tools {
            copied.by_tool.get_mut(&t.name).unwrap().complete =
                (0..4).map(|i| format!("{} {i}", t.description)).collect();
        }
        let (r, ok) = overlap_report(&cats, &copied, 0.35);
        assert!(!ok, "{r}");
        let mut para = phrasings(&cats);
        let lines = [
            "what breaks downstream if I change parse_header?",
            "anything ready for me to pick up next?",
            "which repos have uncommitted work right now?",
            "locate the function called load_config",
        ];
        for t in &cats[0].tools {
            para.by_tool.get_mut(&t.name).unwrap().complete =
                lines.iter().map(|l| (*l).to_owned()).collect();
        }
        let (r, ok) = overlap_report(&cats, &para, 0.35);
        assert!(ok, "{r}");
    }

    #[test]
    fn an_underspecified_phrasing_must_name_a_required_parameter() {
        let cats = catalog();
        let mut v =
            json!({"schema": PHRASINGS_SCHEMA, "tools": [], "direct": ["a?", "b?", "c?", "d?"]});
        for t in &cats[0].tools {
            let missing = if t.name == "graph_search" {
                "not_a_param"
            } else {
                "target"
            };
            let under: Vec<Value> = if t.params.is_empty() {
                vec![]
            } else {
                vec![
                    json!({"text": format!("u1 {}", t.name), "missing": missing}),
                    json!({"text": format!("u2 {}", t.name), "missing": missing}),
                ]
            };
            v["tools"].as_array_mut().unwrap().push(json!({
                "tool": t.name,
                "complete": (0..4).map(|i| format!("c{i} {}", t.name)).collect::<Vec<_>>(),
                "underspecified": under,
            }));
        }
        let err = parse_phrasings(&v, &cats).err().unwrap_or_default();
        assert!(err.contains("not_a_param"), "{err}");
    }

    fn phrasings_json(ph: &Phrasings) -> Value {
        json!({
            "schema": PHRASINGS_SCHEMA,
            "tools": ph.by_tool.iter().map(|(name, p)| json!({
                "tool": name,
                "complete": p.complete,
                "underspecified": p.underspecified.iter()
                    .map(|(t, m)| json!({"text": t, "missing": m})).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "direct": ph.direct,
        })
    }

    /// A shared tool name whose parameter is required in one catalog and absent in another
    /// (devcouncil's and manvi's `devcouncil_renew_lease`): the file parses, and its
    /// underspecified phrasings make "ask" rows only where the parameter is required. There the
    /// request lacks an argument; in the other catalog it is complete.
    #[test]
    fn a_shared_tool_asks_only_where_its_parameter_is_required() {
        let mut cats = catalog();
        let mut other = cats[0].clone();
        other.app = "other".to_owned();
        let gs = other
            .tools
            .iter_mut()
            .find(|t| t.name == "graph_search")
            .unwrap();
        gs.params.clear();
        cats.push(other);
        let ph = parse_phrasings(&phrasings_json(&phrasings(&cats)), &cats).unwrap();
        let (ds, report) = drafts_from(&base_cfg(), &tc(), &cats, &ph).unwrap();
        let asks = |app: &str| {
            ds.iter()
                .filter(|d| {
                    d.template.starts_with("ask:graph_search:")
                        && d.stratum.starts_with(&format!("{FAMILY}/{app}/ask"))
                })
                .count()
        };
        assert!(asks("demo") > 0);
        assert_eq!(asks("other"), 0);
        assert_eq!(report["ask_phrasings_skipped_not_required_in_catalog"], 2);
    }

    /// A catalog where the tool requires a parameter that no underspecified phrasing names has no
    /// "ask" rows for it, and is refused even though every phrasing is valid somewhere.
    #[test]
    fn a_catalog_whose_required_parameter_no_phrasing_names_is_refused() {
        let mut cats = catalog();
        let mut other = cats[0].clone();
        other.app = "other".to_owned();
        for p in &mut other
            .tools
            .iter_mut()
            .find(|t| t.name == "graph_search")
            .unwrap()
            .params
        {
            p.name = "limit".to_owned();
        }
        cats.push(other);
        let err = parse_phrasings(&phrasings_json(&phrasings(&cats)), &cats)
            .err()
            .unwrap_or_default();
        assert!(
            err.contains("graph_search (catalog other)") && err.contains("at least 2"),
            "{err}"
        );
    }

    /// An excluded tool is never offered and never gold; an exclusion naming no catalog tool is
    /// refused rather than silently ignored.
    #[test]
    fn an_excluded_tool_never_appears_and_a_stale_exclusion_is_refused() {
        let cats = catalog();
        let ph = phrasings(&cats);
        let ex: BTreeMap<String, String> = [("repo_status".to_owned(), "test".to_owned())].into();
        let (cats2, ph2) = exclude(cats.clone(), ph.clone(), &ex).unwrap();
        let (ds, _) = drafts_from(&base_cfg(), &tc(), &cats2, &ph2).unwrap();
        assert!(!ds.is_empty());
        for d in &ds {
            assert!(
                !d.context.contains("repo_status") && !d.options.iter().any(|o| o == "repo_status")
            );
        }
        let stale: BTreeMap<String, String> =
            [("no_such_tool".to_owned(), "test".to_owned())].into();
        let err = exclude(cats, ph, &stale).err().unwrap_or_default();
        assert!(err.contains("no_such_tool"), "{err}");
    }

    /// The shipped config, catalog and phrasings, end to end: pins match, the files parse, the
    /// excluded tools never appear, the drafts assemble under the split rules.
    #[test]
    fn the_shipped_tool_data_generate_and_assemble() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/synth/tools-v6-2026-10-06.json");
        let (c, _) = Config::load(&path).unwrap();
        let g = generate(&c).unwrap();
        let excluded = &c.tools.as_ref().unwrap().exclude_tools;
        assert!(!excluded.is_empty());
        for d in &g.drafts {
            for name in excluded.keys() {
                assert!(!d.context.contains(&format!("\"{name}\"")) && !d.options.contains(name));
            }
        }
        assert!(
            g.report["ask_phrasings_skipped_not_required_in_catalog"]
                .as_u64()
                .unwrap()
                > 0
        );
        assemble(&c, &g.drafts).unwrap();
    }

    #[test]
    fn a_one_member_group_is_refused() {
        let v = json!({"schema": CATALOG_SCHEMA, "catalogs": [{"app": "x", "tools": [
            {"name": "a", "description": "d", "params": [], "group": "solo"},
            {"name": "b", "description": "d", "params": [], "group": null}]}]});
        let err = parse_catalogs(&v).err().unwrap_or_default();
        assert!(err.contains("one member"), "{err}");
    }

    /// Privacy: the shipped data files name no person and no home path (the human's
    /// "Synthetic only" ruling; the owner's handle is replaced at extraction).
    #[test]
    fn the_shipped_data_files_carry_no_personal_data() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/synth");
        let mut checked = 0;
        for name in [
            "tool-catalog-v6-2026-10-06.json",
            "tool-phrasings-v6-2026-10-06.json",
        ] {
            let text =
                std::fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let lower = text.to_lowercase();
            assert!(!lower.contains("bharath"), "{name} names the owner");
            assert!(!text.contains("/Users/"), "{name} holds a home path");
            checked += 1;
        }
        assert_eq!(checked, 2);
    }
}
