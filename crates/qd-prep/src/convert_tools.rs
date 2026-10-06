//! The tool-selection pool: When2Call train and ToolACE, one slot shape (plan section 4, "Tool
//! selection"): a choice over the row's tool names plus three fixed actions -- ask a clarifying
//! question, say it cannot help with these tools, answer directly.
//!
//! One family per source (`when2call.tool_select`, `toolace.tool_select`): the Python loader maps
//! each family to exactly one source id (`qd_data.decisions._FAMILY_SOURCE`).
//!
//! **When2Call** (cc-by-4.0, "ready for commercial use"; the generating model is not named on the
//! card). Its train files carry no label field: `train_sft` rows are `{tools, messages}` and
//! `train_pref` rows add `chosen_response`/`rejected_response`. The gold is read from the
//! expected (sft) or chosen (pref) response by [`response_form`]: a `<TOOLCALL>` block with
//! exactly one call to a listed tool, else a clarifying question, else an inability statement.
//! A response the rule cannot place on exactly one side is refused (`response_form_ambiguous`),
//! counted. Measured 2026-10-06: sft holds no tool calls and no direct answers; pref holds 3,000
//! single calls. "Answer directly" is never gold in train -- the manifest's `gold_class` shows
//! it. Rows of one prompt (tools + user message) share a group, so sft and pref copies of it
//! never straddle the split.
//!
//! **ToolACE** (apache-2.0; the generating LLM is undisclosed, so its provider-terms risk is
//! unknown). The tool list is the JSON array after the system prompt's marker. v1 takes
//! two-turn rows (user, assistant) whose answer is exactly one call to a listed tool; a
//! multi-call answer is a different slot shape and is counted (`multi_call`), as is a text answer
//! (`not_a_call`: not relabelled). Calls are counted by bracket depth, quote-aware, because
//! names hold spaces and `/`. **Irrelevance by construction** (the xlam-irrelevance *method*,
//! not its data): for a seeded fraction of admitted rows a second row drops the gold tool from
//! the list and its gold is "cannot help". It shares the parent's group. Caveat in the manifest:
//! another listed tool may still serve the request, so these labels can be noisy.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::convert::{self, Acc, Config, PoolFacts, Row, Views};
use crate::decisions::Gold;
use crate::sha256::sha256_hex;

pub const WHEN2CALL: &str = "nvidia/When2Call";
pub const TOOLACE: &str = "Team-ACE/ToolACE";
pub const W2C_FAMILY: &str = "when2call.tool_select";
pub const TOOLACE_FAMILY: &str = "toolace.tool_select";
pub const W2C_FILES: [(&str, &str); 2] = [
    ("train/when2call_train_sft.jsonl", "sft"),
    ("train/when2call_train_pref.jsonl", "pref"),
];
pub const TOOLACE_FILE: &str = "data.json";

pub const ASK: &str = "Ask the user a clarifying question";
pub const UNABLE: &str = "Say it cannot help with the available tools";
pub const DIRECT: &str = "Answer directly without calling a tool";
pub const QUESTION: &str = "What should the assistant do next: call one of the listed tools (by \
     name), ask the user a clarifying question, say it cannot help with the available tools, or \
     answer directly?";
const TOOLACE_MARKER: &str = "Here is a list of functions in JSON format that you can invoke:";

/// The licence notes every row of each source carries in the manifest.
pub fn licence_notes() -> BTreeMap<&'static str, String> {
    BTreeMap::from([
        (
            WHEN2CALL,
            "cc-by-4.0 (card: \"ready for commercial use\"); attribution: Ross et al., \
                     When2Call, NAACL 2025, NVIDIA. The generating model is not named on the card. \
                     Train splits only; test_mcq and test_llm_judge are decontamination targets"
                .to_owned(),
        ),
        (
            TOOLACE,
            "apache-2.0; attribution: Team-ACE, ToolACE (arXiv 2409.00920). The generating \
                   LLM is undisclosed: provider-terms risk unknown"
                .to_owned(),
        ),
    ])
}

/// What an assistant response is, by the When2Call rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Form {
    Call(String),
    Ask,
    Unable,
}

const UNABLE_MARKS: [&str; 11] = [
    "unable",
    "can't",
    "cannot",
    "can not",
    "not able to",
    "beyond my capabilities",
    "don't have the capability",
    "don't have the ability",
    "don't have access",
    "i'm sorry",
    "apologi",
];
const ASK_MARKS: [&str; 7] = [
    "could you",
    "please provide",
    "please specify",
    "please tell",
    "i need to know",
    "i'll need",
    "i need the",
];

/// The When2Call response rule. A `<TOOLCALL>[...]</TOOLCALL>` block must hold exactly one call
/// (`multi_call` otherwise). Text is a question when it has a `?` or an asking phrase and no
/// inability phrase, an inability statement when the reverse; both or neither is ambiguous.
pub fn response_form(text: &str) -> Result<Form, &'static str> {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("<TOOLCALL>") {
        let inner = rest
            .strip_suffix("</TOOLCALL>")
            .ok_or("malformed_toolcall")?;
        let calls: Value = serde_json::from_str(inner).map_err(|_| "malformed_toolcall")?;
        let calls = calls.as_array().ok_or("malformed_toolcall")?;
        return match calls.as_slice() {
            [one] => one
                .get("name")
                .and_then(Value::as_str)
                .map(|n| Form::Call(n.to_owned()))
                .ok_or("malformed_toolcall"),
            [] => Err("malformed_toolcall"),
            _ => Err("multi_call"),
        };
    }
    let lower = t.to_lowercase();
    let unable = UNABLE_MARKS.iter().any(|m| lower.contains(m));
    let ask = lower.contains('?') || ASK_MARKS.iter().any(|m| lower.contains(m));
    match (ask, unable) {
        (true, false) => Ok(Form::Ask),
        (false, true) => Ok(Form::Unable),
        _ => Err("response_form_ambiguous"),
    }
}

/// A tool's name from its JSON object.
fn tool_name(tool: &Value) -> Option<&str> {
    tool.get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.trim().is_empty())
}

/// The context: each tool's spec on one line, then the user's message.
fn render(tools: &[Value], user: &str) -> String {
    let mut s = String::from("Available tools:\n");
    if tools.is_empty() {
        s.push_str("(none)\n");
    }
    for t in tools {
        s.push_str(&t.to_string());
        s.push('\n');
    }
    s.push_str("\nUser: ");
    s.push_str(user);
    s
}

/// The options in canonical order (tool names, then the three actions) and the gold's index.
fn options(tools: &[Value], gold: &Form) -> Result<(Vec<String>, usize), &'static str> {
    let mut opts: Vec<String> = Vec::with_capacity(tools.len() + 3);
    for t in tools {
        opts.push(tool_name(t).ok_or("tool_without_name")?.to_owned());
    }
    let n = opts.len();
    opts.extend([ASK, UNABLE, DIRECT].map(str::to_owned));
    let g = match gold {
        Form::Call(name) => opts[..n]
            .iter()
            .position(|o| o == name)
            .ok_or("gold_not_in_tools")?,
        Form::Ask => n,
        Form::Unable => n + 1,
    };
    Ok((opts, g))
}

fn class(f: &Form) -> &'static str {
    match f {
        Form::Call(_) => "call",
        Form::Ask => "ask",
        Form::Unable => "unable",
    }
}

/// One When2Call train row.
pub fn when2call_row(kind: &str, line: usize, r: &Value) -> Result<Row, String> {
    let tools: Vec<Value> = r
        .get("tools")
        .and_then(Value::as_array)
        .ok_or("malformed")?
        .iter()
        .map(|t| {
            t.as_str()
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("malformed_tools")?;
    let messages = r
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("malformed")?;
    let (user, response) = match (kind, messages.as_slice()) {
        ("sft", [u, a]) => (u, a),
        ("pref", [u]) => (u, r.get("chosen_response").ok_or("malformed")?),
        _ => return Err("multi_turn".into()),
    };
    if user.get("role").and_then(Value::as_str) != Some("user")
        || response.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return Err("malformed_roles".into());
    }
    let user = user
        .get("content")
        .and_then(Value::as_str)
        .ok_or("malformed")?;
    let form = response_form(
        response
            .get("content")
            .and_then(Value::as_str)
            .ok_or("malformed")?,
    )?;
    let (opts, gold) = options(&tools, &form)?;
    let context = render(&tools, user);
    // Source-prefixed: the two sources split under their own scopes, so a prompt both carry
    // must not share one key across scopes (it could land on both sides of the split).
    let group = format!("when2call:{}", &sha256_hex(context.as_bytes())[..16]);
    Ok(Row {
        id: format!("when2call:{kind}:{line}"),
        source_id: WHEN2CALL,
        family_id: W2C_FAMILY,
        stratum: format!("{W2C_FAMILY}/{}/{kind}", class(&form)),
        group_key: group,
        licence: Some("cc-by-4.0".to_owned()),
        context,
        question: QUESTION.to_owned(),
        slot_name: "action",
        options: opts,
        gold: Gold::Option(gold),
        label_basis: if matches!(form, Form::Call(_)) {
            "tool_call_parsed"
        } else {
            "response_form_rule"
        },
    })
}

/// The tool list after the ToolACE system prompt's marker.
pub fn toolace_tools(system: &str) -> Result<Vec<Value>, &'static str> {
    let at = system.find(TOOLACE_MARKER).ok_or("system_unparsed")?;
    let rest = &system[at + TOOLACE_MARKER.len()..];
    let open = rest.find('[').ok_or("system_unparsed")?;
    let mut it = serde_json::Deserializer::from_str(&rest[open..]).into_iter::<Value>();
    match it.next() {
        Some(Ok(Value::Array(tools))) if tools.iter().all(|t| tool_name(t).is_some()) => Ok(tools),
        _ => Err("system_unparsed"),
    }
}

/// The names of the calls in a ToolACE answer `[f(a=1), g(b="x, y")]`, split at depth 0 and
/// outside quotes. `Err("not_a_call")` for text that is not a bracketed call list.
pub fn toolace_calls(answer: &str) -> Result<Vec<String>, &'static str> {
    let t = answer.trim();
    let inner = t
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or("not_a_call")?;
    let mut names = Vec::new();
    let mut name = String::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut in_args = false;
    for c in inner.chars() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        if in_args {
            match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    depth = depth.checked_sub(1).ok_or("malformed_call")?;
                    if depth == 0 {
                        in_args = false;
                    }
                }
                _ => {}
            }
            continue;
        }
        match c {
            '(' => {
                let n = name.trim();
                if n.is_empty() {
                    return Err("malformed_call");
                }
                names.push(n.to_owned());
                name.clear();
                in_args = true;
                depth = 1;
            }
            ',' if name.trim().is_empty() => {}
            ',' => return Err("malformed_call"),
            _ => name.push(c),
        }
    }
    if in_args || quote.is_some() || !name.trim().is_empty() || names.is_empty() {
        return Err("malformed_call");
    }
    Ok(names)
}

/// One ToolACE row, and its irrelevance row when the seeded draw picks it.
pub fn toolace_rows(
    cfg: &Config,
    irrelevance: f64,
    idx: usize,
    r: &Value,
) -> Result<Vec<Row>, String> {
    let tools = toolace_tools(r.get("system").and_then(Value::as_str).ok_or("malformed")?)?;
    let conv = r
        .get("conversations")
        .and_then(Value::as_array)
        .ok_or("malformed")?;
    let [u, a] = conv.as_slice() else {
        return Err("multi_turn".into());
    };
    let role = |m: &Value| m.get("from").and_then(Value::as_str).map(str::to_owned);
    if role(u).as_deref() != Some("user") || role(a).as_deref() != Some("assistant") {
        return Err("malformed_roles".into());
    }
    let user = u.get("value").and_then(Value::as_str).ok_or("malformed")?;
    let calls = toolace_calls(a.get("value").and_then(Value::as_str).ok_or("malformed")?)?;
    let [gold_name] = calls.as_slice() else {
        return Err("multi_call".into());
    };
    let form = Form::Call(gold_name.clone());
    let (opts, gold) = options(&tools, &form)?;
    let context = render(&tools, user);
    let group = format!("toolace:{}", &sha256_hex(context.as_bytes())[..16]);
    let base = Row {
        id: format!("toolace:{idx}"),
        source_id: TOOLACE,
        family_id: TOOLACE_FAMILY,
        stratum: format!("{TOOLACE_FAMILY}/call"),
        group_key: group.clone(),
        licence: Some("apache-2.0".to_owned()),
        context,
        question: QUESTION.to_owned(),
        slot_name: "action",
        options: opts,
        gold: Gold::Option(gold),
        label_basis: "tool_call_parsed",
    };
    let mut out = vec![base];
    if crate::decisions::unit_draw(cfg.seed, &["toolace-irrelevance", &idx.to_string()])
        < irrelevance
    {
        let kept: Vec<Value> = tools
            .iter()
            .filter(|t| tool_name(t) != Some(gold_name.as_str()))
            .cloned()
            .collect();
        let (opts, gold) = options(&kept, &Form::Unable)?;
        out.push(Row {
            id: format!("toolace:{idx}:irrelevant"),
            stratum: format!("{TOOLACE_FAMILY}/irrelevance"),
            context: render(&kept, user),
            options: opts,
            gold: Gold::Option(gold),
            label_basis: "irrelevance_by_construction",
            group_key: group,
            ..out[0].clone()
        });
    }
    Ok(out)
}

/// The tool pool's decontamination targets: When2Call's test splits and every BFCL file.
pub fn target_files(views: &Views) -> Vec<(String, String, String)> {
    let mut out = vec![
        (
            "when2call-test-mcq".to_owned(),
            WHEN2CALL.to_owned(),
            "test/when2call_test_mcq.jsonl".to_owned(),
        ),
        (
            "when2call-test-llm-judge".to_owned(),
            WHEN2CALL.to_owned(),
            "test/when2call_test_llm_judge.jsonl".to_owned(),
        ),
    ];
    for e in &views.entries {
        if e.dataset == "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
            && e.file.ends_with(".json")
        {
            out.push((
                format!("bfcl/{}", e.file),
                e.dataset.clone(),
                e.file.clone(),
            ));
        }
    }
    out
}

pub fn build(
    cfg: &Config,
    views: &Views,
    mut digests: BTreeMap<String, String>,
    threads: usize,
) -> Result<crate::decisions::Built, String> {
    let p = cfg.pool("tools")?;
    let irrelevance = convert::fraction(p, "toolace_irrelevance_fraction", "pools.tools")?;
    let mut acc = Acc::new(cfg);
    for (file, kind) in W2C_FILES {
        let v = views.train_rows(WHEN2CALL, file)?;
        let got = convert::read_view(v, |n, r| {
            acc.offer(WHEN2CALL, W2C_FAMILY, when2call_row(kind, n, &r))
        })?;
        digests.insert(format!("{WHEN2CALL}/{file}"), got);
    }
    let mut irrelevance_rows = 0usize;
    let v = views.train_rows(TOOLACE, TOOLACE_FILE)?;
    let got = convert::read_view(v, |n, r| match toolace_rows(cfg, irrelevance, n, &r) {
        Ok(rows) => {
            irrelevance_rows += rows.len() - 1;
            for row in rows {
                acc.offer(TOOLACE, TOOLACE_FAMILY, Ok(row))?;
            }
            Ok(())
        }
        Err(reason) => acc.offer(TOOLACE, TOOLACE_FAMILY, Err(reason)),
    })?;
    digests.insert(format!("{TOOLACE}/{TOOLACE_FILE}"), got);
    let names = target_files(views);
    let refs: Vec<(&str, &str, &str)> = names
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let (targets, target_digests) = views.target_texts(&refs)?;
    digests.extend(target_digests);
    let facts = PoolFacts {
        name: "tools",
        licence_notes: licence_notes(),
        caps: json!({"toolace_irrelevance_fraction": irrelevance,
                     "toolace_irrelevance_rows_offered": irrelevance_rows,
                     "per_row_caps": "none: every admitted row is a candidate; >13 tools is refused \
                                      as too_many_options (16 options at most), not subsampled"}),
        id_checks: json!({"state": "not_applicable",
                          "reason": "When2Call test and BFCL carry no ids shared with the train files; \
                                     text containment covers them"}),
        notes: json!({
            "when2call_label_rule": "train files carry no label field; gold from the expected (sft) \
                or chosen (pref) response: one <TOOLCALL> -> that tool; else a question -> ask; else \
                an inability statement -> unable; both or neither -> refused response_form_ambiguous",
            "when2call_direct_answer": "never gold in train (sft: ask/unable only; pref: call/ask/unable); \
                see gold_class",
            "toolace_scope": "v1: two-turn rows answered by exactly one call; multi_call, not_a_call \
                (text answers, not relabelled) and multi_turn are counted refusals",
            "toolace_irrelevance_caveat": "the gold tool is removed and the gold becomes 'cannot help'; \
                another listed tool may still serve the request (label noise unmeasured)",
            "rule_7": "options are classifications of the next action; Lappi admits, it never \
                authorises a call",
        }),
    };
    convert::build(acc, targets, digests, facts, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools_json(names: &[&str]) -> Vec<Value> {
        names
            .iter()
            .map(|n| {
                json!(format!(
                    "{{\"name\": \"{n}\", \"description\": \"d\", \"parameters\": {{}}}}"
                ))
            })
            .collect()
    }

    #[test]
    fn when2call_gold_comes_from_the_response_form() {
        let sft = |content: &str| {
            json!({"tools": tools_json(&["get_weather", "get_time"]),
                   "messages": [{"role": "user", "content": "What is the weather?"},
                                {"role": "assistant", "content": content}]})
        };
        let ask = when2call_row(
            "sft",
            1,
            &sft("To proceed, could you please provide the city?"),
        )
        .unwrap();
        assert_eq!(
            ask.options[match ask.gold {
                Gold::Option(i) => i,
                Gold::Noul => panic!(),
            }],
            ASK
        );
        assert_eq!(ask.label_basis, "response_form_rule");
        let unable = when2call_row(
            "sft",
            2,
            &sft("Apologies, but I'm unable to provide real-time data."),
        )
        .unwrap();
        assert_eq!(
            unable.options[match unable.gold {
                Gold::Option(i) => i,
                Gold::Noul => panic!(),
            }],
            UNABLE
        );
        assert_eq!(
            when2call_row(
                "sft",
                3,
                &sft("I'm sorry, I can't. Could you provide more details?")
            )
            .unwrap_err(),
            "response_form_ambiguous"
        );
        let pref = json!({"tools": tools_json(&["get_weather"]),
            "messages": [{"role": "user", "content": "Weather in Paris"}],
            "chosen_response": {"role": "assistant",
                "content": "<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]</TOOLCALL>"},
            "rejected_response": {"role": "assistant", "content": "x"}});
        let call = when2call_row("pref", 4, &pref).unwrap();
        assert_eq!(
            call.options[match call.gold {
                Gold::Option(i) => i,
                Gold::Noul => panic!(),
            }],
            "get_weather"
        );
        assert_eq!(call.label_basis, "tool_call_parsed");
        let mut two = pref.clone();
        two["chosen_response"]["content"] = json!(
            "<TOOLCALL>[{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\", \"arguments\": {}}]</TOOLCALL>"
        );
        assert_eq!(when2call_row("pref", 5, &two).unwrap_err(), "multi_call");
        let mut absent = pref.clone();
        absent["chosen_response"]["content"] =
            json!("<TOOLCALL>[{\"name\": \"nope\", \"arguments\": {}}]</TOOLCALL>");
        assert_eq!(
            when2call_row("pref", 6, &absent).unwrap_err(),
            "gold_not_in_tools"
        );
    }

    #[test]
    fn sft_and_pref_copies_of_one_prompt_share_a_group() {
        let tools = tools_json(&["f"]);
        let sft = json!({"tools": tools, "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "Could you please provide x?"}]});
        let pref = json!({"tools": tools, "messages": [{"role": "user", "content": "q"}],
                          "chosen_response": {"role": "assistant", "content": "Could you please provide x?"}});
        assert_eq!(
            when2call_row("sft", 1, &sft).unwrap().group_key,
            when2call_row("pref", 9, &pref).unwrap().group_key
        );
    }

    #[test]
    fn toolace_calls_are_split_at_depth_zero_outside_quotes() {
        assert_eq!(
            toolace_calls("[Market Trends API(trend_type=\"MARKET_INDEXES\", country=\"us\")]")
                .unwrap(),
            ["Market Trends API"]
        );
        assert_eq!(
            toolace_calls("[/madlibs-diceware(nphrase=2)]").unwrap(),
            ["/madlibs-diceware"]
        );
        assert_eq!(
            toolace_calls("[GetCompetitions()]").unwrap(),
            ["GetCompetitions"]
        );
        assert_eq!(
            toolace_calls("[f(a=\"x), g(\", b=[1, (2)]), g(c='it\\'s')]").unwrap(),
            ["f", "g"]
        );
        assert_eq!(
            toolace_calls("Could you tell me the season?").unwrap_err(),
            "not_a_call"
        );
        assert_eq!(toolace_calls("[f(a=1]").unwrap_err(), "malformed_call");
    }

    fn toolace_row(answer: &str) -> Value {
        json!({"system": format!("You are an expert.\n{TOOLACE_MARKER}\n[{{\"name\": \"Weather API\", \
                \"description\": \"w\"}}, {{\"name\": \"Time API\", \"description\": \"t\"}}]. \nShould you decide..."),
               "conversations": [{"from": "user", "value": "Weather in Rome?"},
                                 {"from": "assistant", "value": answer}]})
    }

    #[test]
    fn toolace_single_calls_only_and_the_irrelevance_row_drops_the_gold_tool() {
        let mut cfg = crate::convert::tests::cfg();
        cfg.seed = 3;
        let rows =
            toolace_rows(&cfg, 1.0, 0, &toolace_row("[Weather API(city=\"Rome\")]")).unwrap();
        assert_eq!(rows.len(), 2);
        let g = |r: &Row| {
            r.options[match r.gold {
                Gold::Option(i) => i,
                Gold::Noul => panic!(),
            }]
            .clone()
        };
        assert_eq!(g(&rows[0]), "Weather API");
        assert_eq!(g(&rows[1]), UNABLE);
        assert!(!rows[1].options.iter().any(|o| o == "Weather API"));
        assert!(!rows[1].context.contains("Weather API"));
        assert_eq!(rows[0].group_key, rows[1].group_key);
        assert_eq!(rows[1].label_basis, "irrelevance_by_construction");
        assert_eq!(
            toolace_rows(
                &cfg,
                1.0,
                1,
                &toolace_row("[Weather API(city=\"Rome\"), Time API()]")
            )
            .unwrap_err(),
            "multi_call"
        );
        assert_eq!(
            toolace_rows(&cfg, 1.0, 2, &toolace_row("Which city?")).unwrap_err(),
            "not_a_call"
        );
        let none = toolace_rows(&cfg, 0.0, 3, &toolace_row("[Time API()]")).unwrap();
        assert_eq!(none.len(), 1);
        let bad = json!({"system": "no marker", "conversations": []});
        assert_eq!(
            toolace_rows(&cfg, 0.5, 4, &bad).unwrap_err(),
            "system_unparsed"
        );
    }
}
