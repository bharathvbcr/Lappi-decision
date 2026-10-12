//! The v6 caller families (card ft-2d0a5e4b27e): `gitpulse.commit_type` and `devtype.route`
//! rendered byte-identical to what each app's own request builder sends
//! (`docs/caller-contract.md` section 4).
//!
//! The pinned fixture `fixtures/caller-requests-v6.json` holds each app's own builder output,
//! read from the app repository (DevType 24cc8d8, GitPulse 211cd903); nothing in it was written
//! by a Lappi builder. The tests compare what the Lappi builders render against it, and against
//! section 4's table.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use qd_prep::commit_type::{self, KNOWN_TYPES};
use qd_prep::decisions::BARE_CONTEXT_FAMILIES;
use qd_prep::own_swift::{TrainRepo, View};
use qd_prep::synth::{self, Config};
use qd_prep::synth_tools::{
    ROUTE_FAMILY, ROUTE_MAX_QUERY_CHARS, ROUTE_OPTIONS, ROUTE_QUESTION, ROUTE_SLOT,
    route_date_accepts,
};
use serde_json::{Value, json};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qd-prep-caller-{}-{name}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch");
    }
    fs::create_dir_all(&dir).expect("make scratch");
    dir
}

/// The app's request, parsed from its own literal line in the pinned fixture.
fn app_request(app: &str) -> Value {
    let doc: Value = serde_json::from_slice(
        &fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/caller-requests-v6.json"))
            .expect("the pinned fixture"),
    )
    .expect("fixture JSON");
    let a = doc["apps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["app"] == app)
        .unwrap_or_else(|| panic!("no {app} in the fixture"));
    assert!(
        a["commit"].as_str().is_some_and(|c| c.len() == 40),
        "the fixture names the app commit it was read at"
    );
    serde_json::from_str(a["line"].as_str().expect("line")).expect("the app's line is JSON")
}

fn options_of(req: &Value) -> Vec<String> {
    req["slots"][0]["options"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn the_route_builder_renders_the_apps_own_question_slot_and_options() {
    let app = app_request("devtype");
    assert_eq!(app["task"], ROUTE_FAMILY);
    assert_eq!(app["question"], ROUTE_QUESTION);
    assert_eq!(app["slots"][0]["name"], ROUTE_SLOT);
    assert_eq!(app["slots"][0]["type"], "choice");
    assert_eq!(options_of(&app), ROUTE_OPTIONS.map(str::to_owned));
}

#[test]
fn the_commit_type_builder_renders_the_apps_own_question_slot_and_options() {
    let app = app_request("gitpulse");
    assert_eq!(app["task"], commit_type::FAMILY);
    assert_eq!(app["question"], commit_type::QUESTION);
    assert_eq!(app["slots"][0]["name"], commit_type::SLOT);
    assert_eq!(app["slots"][0]["type"], "choice");
    assert_eq!(options_of(&app), KNOWN_TYPES.map(str::to_owned));
}

#[test]
fn contract_section_4_pins_the_same_strings() {
    let doc = fs::read_to_string(repo_root().join("docs/caller-contract.md")).unwrap();
    let section = doc
        .split("## 4. Ask only trained shapes")
        .nth(1)
        .and_then(|s| s.split("## 5.").next())
        .expect("section 4");
    for (task, question, slot, options) in [
        (
            commit_type::FAMILY,
            commit_type::QUESTION,
            commit_type::SLOT,
            KNOWN_TYPES.join(", "),
        ),
        (
            ROUTE_FAMILY,
            ROUTE_QUESTION,
            ROUTE_SLOT,
            ROUTE_OPTIONS.join(", "),
        ),
    ] {
        let row = section
            .lines()
            .find(|l| l.contains(&format!("`{task}`")))
            .unwrap_or_else(|| panic!("no section 4 row for {task}"));
        assert!(row.contains(&format!("`{question}`")), "{task}: question differs in {row}");
        assert!(
            row.contains(&format!("`{slot}` choice `[{options}]`")),
            "{task}: slot or options differ in {row}"
        );
    }
}

#[test]
fn both_families_render_as_their_bare_context() {
    for f in [commit_type::FAMILY, ROUTE_FAMILY] {
        assert!(BARE_CONTEXT_FAMILIES.contains(&f), "{f} is not a bare-context family");
    }
}

// ---------------------------------------------------------------------------------------------
// devtype.route

#[test]
fn the_date_grammar_is_the_palettes() {
    for ok in [
        "date+5", "date-12", "date +3d", "DATE-4D", "date  +  7", "+3d", "-12d", "+ 4 d", "+2w",
        "-1w", "+1d+3h", "+1w+2d", "+1y", "+30m", "+45s", "next friday", "next Monday date",
        "next full date", "full date tomorrow", "  +5d  ", "+1D",
    ] {
        assert!(route_date_accepts(ok), "{ok:?} should be a date");
    }
    for no in [
        "", "date+0", "+0d", "+1d-1d", "date+1000", "date+5x", "+5", "5d", "date three weeks out",
        "next someday", "next friday!", "= 1+2", "> make it polite", "run the standup macro",
        "what is the capital of Australia", "+1234567d", "date+", "date+٣",
    ] {
        assert!(!route_date_accepts(no), "{no:?} should not be a date");
    }
}

/// A `qd-synth-config/v1` for `kind route` over `phrasings`, targets pinned to `target_sha`.
fn route_config(dir: &Path, phrasings: &Value, target_sha: Option<&str>) -> PathBuf {
    let mut cfg: Value =
        serde_json::from_slice(&fs::read(repo_root().join("data/synth/route-v6-2026-10-12.json")).unwrap())
            .unwrap();
    let bytes = serde_json::to_vec(phrasings).unwrap();
    fs::write(dir.join("phrasings.json"), &bytes).unwrap();
    cfg["route"]["phrasings"] = json!("phrasings.json");
    cfg["route"]["phrasings_sha256"] = json!(qd_prep::sha256::sha256_hex(&bytes));
    cfg["rows_per_template"] = json!(3);
    if let Some(sha) = target_sha {
        cfg["target_sha256"] = json!({"route-heldout": sha});
    }
    let path = dir.join("config.json");
    fs::write(&path, serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
    path
}

fn phrasings_with(extra: Vec<(&str, &str, &str)>) -> Value {
    let mut doc: Value = serde_json::from_slice(
        &fs::read(repo_root().join("data/synth/devtype-route-phrasings-v6-2026-10-12.json")).unwrap(),
    )
    .unwrap();
    let t = doc["templates"].as_array_mut().unwrap();
    for (id, class, text) in extra {
        t.push(json!({"id": id, "class": class, "text": text}));
    }
    doc
}

fn generate_err(dir: &Path, phrasings: &Value) -> String {
    let path = route_config(dir, phrasings, None);
    let (cfg, _) = Config::load(&path).expect("config loads");
    match synth::generate(&cfg) {
        Err(e) => e,
        Ok(_) => panic!("the generator accepted a template it must refuse"),
    }
}

#[test]
fn none_comes_only_from_the_unable_and_direct_classes() {
    let d = scratch("none-classes");
    // A date phrase filed under none_unable resolves to resolveDate: refused, not relabelled.
    let e = generate_err(&d, &phrasings_with(vec![("bad-1", "none_unable", "date+5")]));
    assert!(e.contains("bad-1") && e.contains("resolveDate"), "{e}");
    // A class outside the five cannot carry `none`.
    let e = generate_err(&d, &phrasings_with(vec![("bad-2", "none", "what is a comet")]));
    assert!(e.contains("class") && e.contains("none_unable"), "{e}");
    // A plain question filed as a date is not one.
    let e = generate_err(&d, &phrasings_with(vec![("bad-3", "resolveDate", "what is a comet")]));
    assert!(e.contains("bad-3") && e.contains("none"), "{e}");
}

#[test]
fn a_query_two_tools_would_answer_is_refused() {
    let d = scratch("two-tools");
    let e = generate_err(&d, &phrasings_with(vec![("bad-4", "runMacro", "run {macro} macro and paste the {snippet} snippet")]));
    assert!(e.contains("more than one tool"), "{e}");
}

#[test]
fn a_template_id_cannot_be_used_twice_and_a_padded_query_is_refused() {
    let d = scratch("ids");
    let e = generate_err(&d, &phrasings_with(vec![("date-01", "resolveDate", "date+9")]));
    assert!(e.contains("used twice"), "{e}");
    let e = generate_err(&d, &phrasings_with(vec![("bad-5", "resolveDate", "date+9 ")]));
    assert!(e.contains("trimmed"), "{e}");
}

/// The pool built from the committed config and data, end to end.
fn built_route_pool(name: &str) -> (PathBuf, Vec<Value>, Value) {
    let d = scratch(name);
    let cfg_path = route_config(&d, &phrasings_with(vec![]), None);
    let heldout = d.join("heldout").join("route");
    fs::create_dir_all(heldout.parent().unwrap()).unwrap();
    synth::emit_heldout(&cfg_path, &heldout).expect("held-out set");
    let sha = qd_prep::sha256::sha256_hex(&fs::read(heldout.join("targets.jsonl")).unwrap());
    let cfg_path = route_config(&d, &phrasings_with(vec![]), Some(&sha));
    let out = d.join("pool");
    let inputs = synth::Inputs {
        config: cfg_path,
        targets: vec![("route-heldout".to_owned(), heldout.join("targets.jsonl"))],
    };
    synth::run(&inputs, &out, 2).expect("the route pool builds");
    let rows = fs::read_to_string(out.join("examples.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let manifest = serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    (out, rows, manifest)
}

#[test]
fn route_rows_are_the_apps_request_and_none_is_only_unable_or_direct() {
    let (_, rows, manifest) = built_route_pool("route-pool");
    let app = app_request("devtype");
    let want_options = options_of(&app);
    assert!(rows.len() > 100, "{} rows", rows.len());
    assert_eq!(manifest["kind"], "route");
    assert_eq!(manifest["allocation"]["state"], "not_applied");
    // The leak probe ran and is recorded; for this family it is a report, never a pass.
    let probe = &manifest["leak_probe"][ROUTE_FAMILY];
    assert_eq!(probe["state"], "ran");
    assert!(probe["passed"].is_null(), "{probe}");
    assert_eq!(
        probe["report_only"]["ruling"],
        "AUDIT/board-sweep-2026-10-12/fable-leak-probe-devtype-route-ruling.md"
    );
    assert_eq!(probe["report_only"]["criterion"], "surface-rule gold");
    assert!(probe["val_accuracy"].as_f64().is_some());
    let mut by_gold: BTreeMap<String, usize> = BTreeMap::new();
    for r in &rows {
        assert_eq!(r["family_id"], ROUTE_FAMILY);
        assert_eq!(r["slot_name"], app["slots"][0]["name"]);
        // The options are the app's, in the app's order, never shuffled.
        let got: Vec<String> = r["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(got, want_options, "row {}", r["id"]);
        // The context is the query and nothing in front of it.
        let ctx = r["context"].as_str().unwrap();
        assert!(!ctx.contains(ROUTE_QUESTION), "{ctx}");
        assert!(!ctx.starts_with('\n') && ctx == ctx.trim(), "{ctx:?}");
        assert!(ctx.chars().count() <= ROUTE_MAX_QUERY_CHARS);
        assert_eq!(r["gold_noul"], false);
        let gold = r["gold_option"].as_str().unwrap();
        *by_gold.entry(gold.to_owned()).or_default() += 1;
        let stratum = r["stratum"].as_str().unwrap();
        if gold == "none" {
            assert!(
                stratum.ends_with("/none_unable") || stratum.ends_with("/none_direct"),
                "none from {stratum}"
            );
        }
        // Re-derive: the date grammar accepts exactly the resolveDate rows.
        assert_eq!(route_date_accepts(ctx), gold == "resolveDate", "{ctx:?} -> {gold}");
    }
    for o in &want_options {
        assert!(by_gold.contains_key(o), "no row with gold {o}: {by_gold:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// gitpulse.commit_type, on git repositories built in a temp directory

fn git(dir: &Path, args: &[&str]) -> Vec<u8> {
    let date = "2026-03-01T12:00:00+00:00";
    let out = Command::new("git")
        .args([
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
            "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main",
            "-c", "core.autocrlf=false",
        ])
        .args(args)
        .current_dir(dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_DATE", date)
        .env("GIT_COMMITTER_DATE", date)
        .output()
        .expect("run git");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

fn commit_file(dir: &Path, file: &str, bytes: &[u8], message: &str) -> String {
    fs::write(dir.join(file), bytes).unwrap();
    git(dir, &["add", file]);
    git(dir, &["commit", "-q", "-m", message]);
    String::from_utf8(git(dir, &["rev-parse", "HEAD"])).unwrap().trim().to_owned()
}

fn view_of(root: &Path, repos: &[(&str, &str)]) -> View {
    let train: Vec<TrainRepo> = repos
        .iter()
        .map(|(name, slug)| {
            let dir = root.join(name);
            TrainRepo {
                repo: (*name).to_owned(),
                slug: (*slug).to_owned(),
                head: String::from_utf8(git(&dir, &["rev-parse", "HEAD"])).unwrap().trim().to_owned(),
            }
        })
        .collect();
    View {
        code_root: root.to_path_buf(),
        all_repos: train.iter().map(|t| t.repo.clone()).collect(),
        train,
        heldout: vec!["held".to_owned()],
        struck: vec![],
    }
}

fn mk_repo(root: &Path, name: &str) -> PathBuf {
    mk_repo_at(root, name, &format!("bharathvbcr/{}", name.rsplit('/').next().unwrap()))
}

fn mk_repo_at(root: &Path, name: &str, slug: &str) -> PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init", "-q"]);
    git(&dir, &["remote", "add", "origin", &format!("https://github.com/{slug}.git")]);
    dir
}

fn no_blobs() -> std::collections::BTreeSet<String> {
    std::collections::BTreeSet::new()
}

#[test]
fn commit_rows_are_the_commits_diff_bytes_with_the_apps_options_and_the_prefix_as_gold() {
    let root = scratch("ct-basic");
    let a = mk_repo(&root, "a");
    let root_sha = commit_file(&a, "x.txt", b"one\n", "feat(core): first\n");
    let second = commit_file(&a, "x.txt", b"one\ntwo\n", "fix!: second\n\nCo-Authored-By: Bot <b@example.com>");
    commit_file(&a, "y.txt", b"y\n", "Update y");
    commit_file(&a, "z.txt", b"z\n", "wip: scratch");
    let view = view_of(&root, &[("a", "bharathvbcr/a")]);
    let (rows, report) = commit_type::candidates(&view, 7, &no_blobs()).unwrap();
    assert_eq!(rows.len(), 2, "{report}");
    let by_gold: BTreeMap<String, &qd_prep::decisions::Candidate> = rows
        .iter()
        .map(|c| {
            let qd_prep::decisions::Gold::Option(i) = c.gold else { panic!("noul gold") };
            (c.options[i].clone(), c)
        })
        .collect();
    // The options are the app's, in its order, on every row.
    let app = app_request("gitpulse");
    for c in &rows {
        assert_eq!(c.options, options_of(&app));
        assert_eq!(c.slot_name, app["slots"][0]["name"].as_str().unwrap());
        assert!(c.question.is_empty());
        assert_eq!(c.rendered_context(), c.context, "bare context");
    }
    // The context is `git diff <parent> <commit>`'s bytes, the empty tree for a root commit.
    let want_fix = git(&a, &["-c", "core.quotepath=false", "diff", "--no-color", "--no-ext-diff", "-U3", &format!("{second}^"), &second]);
    assert_eq!(by_gold["fix"].context.as_bytes(), want_fix.as_slice());
    let want_feat = git(
        &a,
        &["-c", "core.quotepath=false", "diff", "--no-color", "--no-ext-diff", "-U3", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", &root_sha],
    );
    assert_eq!(by_gold["feat"].context.as_bytes(), want_feat.as_slice());
    // Counted, never silent.
    let dropped = &report["per_repo"]["a"]["dropped"];
    assert_eq!(dropped["no_conventional_prefix"], 1);
    assert_eq!(dropped["type_not_in_known_types"], 1);
    assert_eq!(report["per_repo"]["a"]["kept_agent_authored"], 1);
    assert_eq!(report["per_repo"]["a"]["kept_human_authored"], 1);
    assert_eq!(report["per_repo"]["a"]["prefixes_outside_known_types"]["wip"], 1);
}

#[test]
fn gitpulse_is_refused_as_a_producer_and_held_out_repos_are_never_read() {
    let root = scratch("ct-refuse");
    let g = mk_repo(&root, "gp");
    commit_file(&g, "x.txt", b"x\n", "feat: x");
    let view = view_of(&root, &[("gp", "bharathvbcr/gitpulse")]);
    let e = commit_type::candidates(&view, 7, &no_blobs()).unwrap_err();
    assert!(e.contains("held-out eval") && e.contains("refused as a producer"), "{e}");
    // The held-out name in the view points at a path that does not exist: reading it would fail.
    let a = mk_repo(&root, "a");
    commit_file(&a, "x.txt", b"x\n", "feat: x");
    let ok = view_of(&root, &[("a", "bharathvbcr/a")]);
    assert_eq!(ok.heldout, vec!["held".to_owned()]);
    commit_type::candidates(&ok, 7, &no_blobs()).expect("a view with a held-out repo that is absent reads fine");
}

#[test]
fn the_cap_the_bound_and_utf8_each_drop_a_row_and_say_so() {
    let root = scratch("ct-drops");
    let a = mk_repo(&root, "a");
    for i in 0..5 {
        commit_file(&a, &format!("f{i}.txt"), format!("{i}\n").as_bytes(), &format!("chore: add {i}"));
    }
    let big = "x".repeat(qd_prep::decisions::MAX_CONTEXT_BYTES + 10);
    commit_file(&a, "big.txt", big.as_bytes(), "feat: huge");
    commit_file(&a, "bad.txt", b"caf\xe9 not utf-8\n", "fix: latin1");
    commit_file(&a, "same1.txt", b"same\n", "docs: s1");
    let view = view_of(&root, &[("a", "bharathvbcr/a")]);
    let (rows, report) = commit_type::candidates_capped(&view, 7, 3, &no_blobs()).unwrap();
    assert_eq!(rows.len(), 3, "{report}");
    let d = &report["per_repo"]["a"]["dropped"];
    // 8 prefixed commits, 3 kept by the cap, 5 dropped: the cap is applied among the rows that
    // pass every other rule, in a seeded order, so which of the 5 is dropped depends on the draw.
    let total: usize = d.as_object().unwrap().values().map(|v| v.as_u64().unwrap() as usize).sum();
    assert_eq!(total + rows.len(), 8, "{report}");
    let (all, all_report) = commit_type::candidates_capped(&view, 7, 400, &no_blobs()).unwrap();
    assert_eq!(all.len(), 6, "{all_report}");
    let d = &all_report["per_repo"]["a"]["dropped"];
    assert_eq!(d["context_over_bound"], 1);
    assert_eq!(d["context_not_utf8"], 1);
}

#[test]
fn a_pool_is_whole_or_not_at_all_and_its_manifest_says_not_applied() {
    let root = scratch("ct-pool");
    let a = mk_repo(&root, "a");
    for i in 0..30 {
        commit_file(&a, &format!("f{i}.txt"), format!("{i}\n").as_bytes(), &format!("feat: add {i}"));
    }
    let view = view_of(&root, &[("a", "bharathvbcr/a")]);
    let manifest_path = root.join("own-repos.json");
    fs::write(&manifest_path, b"{}").unwrap();
    let (bytes, manifest) = commit_type::build_from_view(&view, &manifest_path, b"{}", 7, &no_blobs()).unwrap();
    assert_eq!(manifest["allocation"]["state"], "not_applied");
    assert_eq!(manifest["decontamination"]["state"], "not_run");
    assert_eq!(manifest["family"], commit_type::FAMILY);
    let lines: Vec<Value> =
        String::from_utf8(bytes).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 30);
    assert!(lines.iter().all(|l| l["licence"] == "owner-granted" && l["source_id"] == commit_type::SOURCE));
    assert!(lines.iter().all(|l| l["split"] == "train" || l["split"] == "val"));
    let empty = scratch("ct-empty");
    let e = mk_repo(&empty, "e");
    commit_file(&e, "x.txt", b"x\n", "Update");
    let none = view_of(&empty, &[("e", "bharathvbcr/e")]);
    assert!(commit_type::build_from_view(&none, &manifest_path, b"{}", 7, &no_blobs()).unwrap_err().contains("nothing"));
}

#[test]
fn a_bare_context_row_with_a_question_is_refused_and_other_families_still_need_one() {
    use qd_prep::decisions::{Candidate, Gold, structural_refusal};
    let row = |family: &str, question: &str| Candidate {
        id: "x".into(),
        source_id: "lappi/synth-route",
        family_id: family.into(),
        stratum: format!("{family}/s"),
        group_key: "g".into(),
        licence: "synthetic-by-rule".into(),
        context: "date+3".into(),
        question: question.into(),
        slot_name: "route".into(),
        options: ROUTE_OPTIONS.map(str::to_owned).to_vec(),
        gold: Gold::Option(0),
        label_basis: "by_construction",
        split: "train",
    };
    assert_eq!(structural_refusal(&row(ROUTE_FAMILY, "")), None);
    assert_eq!(
        structural_refusal(&row(ROUTE_FAMILY, "Which?")),
        Some("question_in_bare_context_family")
    );
    assert_eq!(structural_refusal(&row("apps.tool_select", "")), Some("empty_question"));
    assert_eq!(row(ROUTE_FAMILY, "").rendered_context(), "date+3");
    assert_eq!(row("apps.tool_select", "Q?").rendered_context(), "Q?\n\ndate+3");
}

fn view_with(root: &Path, repos: &[(&str, &str)], all: &[&str], struck: &[(&str, &str)]) -> View {
    let mut v = view_of(root, repos);
    v.all_repos = all.iter().map(|s| (*s).to_owned()).collect();
    v.struck = struck.iter().map(|(r, p)| ((*r).to_owned(), (*p).to_owned())).collect();
    v
}

#[test]
fn a_commit_touching_a_struck_path_or_a_nested_repo_is_dropped_and_counted() {
    let root = scratch("ct-struck");
    let a = mk_repo(&root, "a");
    fs::create_dir_all(a.join("docs/business")).unwrap();
    fs::create_dir_all(a.join("inner")).unwrap();
    fs::create_dir_all(a.join("docs/public")).unwrap();
    commit_file(&a, "docs/business/plan.md", b"secret plan\n", "docs: business plan");
    commit_file(&a, "inner/f.txt", b"nested\n", "feat: nested repo file");
    // One commit touching a struck file among others is refused whole.
    fs::write(a.join("docs/public/ok.md"), b"ok\n").unwrap();
    fs::write(a.join("docs/business/more.md"), b"more\n").unwrap();
    git(&a, &["add", "-A"]);
    git(&a, &["commit", "-q", "-m", "fix: mixed commit"]);
    commit_file(&a, "docs/public/fine.md", b"fine\n", "docs: fine");
    let view = view_with(&root, &[("a", "bharathvbcr/a")], &["a", "a/inner"], &[("a", "docs/business/")]);
    let (rows, report) = commit_type::candidates(&view, 7, &no_blobs()).unwrap();
    let d = &report["per_repo"]["a"]["dropped"];
    assert_eq!(d["struck_path"], 2, "{report}");
    assert_eq!(d["inside_nested_repository"], 1, "{report}");
    assert_eq!(rows.len(), 1);
    assert!(rows.iter().all(|c| !c.context.contains("docs/business/") && !c.context.contains("inner/")));
}

#[test]
fn a_blob_of_a_held_out_repository_is_refused_under_any_path() {
    let root = scratch("ct-heldout-blob");
    let a = mk_repo(&root, "a");
    fs::create_dir_all(a.join("vendored")).unwrap();
    commit_file(&a, "vendored/copy.txt", b"held-out repo content\n", "feat: vendored copy");
    commit_file(&a, "other.txt", b"mine\n", "feat: mine");
    let id = String::from_utf8(git(&a, &["hash-object", "vendored/copy.txt"])).unwrap().trim().to_owned();
    let view = view_of(&root, &[("a", "bharathvbcr/a")]);
    let blobs: std::collections::BTreeSet<String> = [id].into();
    let (rows, report) = commit_type::candidates(&view, 7, &blobs).unwrap();
    assert_eq!(report["per_repo"]["a"]["dropped"]["heldout_repository_content"], 1, "{report}");
    assert_eq!(rows.len(), 1);
}

#[test]
fn a_credential_in_a_diff_drops_the_commit_and_a_word_boundary_miss_does_not() {
    let root = scratch("ct-secret");
    let a = mk_repo(&root, "a");
    commit_file(&a, "env.txt", b"apiKey: \"AIzaSyA7abcdefghijklmnopqrstuvwxyz012345\"\n", "chore: firebase env");
    commit_file(&a, "word.txt", b"TASKAKIA0123456789012345 is a task id\n", "chore: not a key");
    commit_file(&a, "late.txt", b"see AKIA short then AKIAABCDEFGHIJKLMNOP\n", "chore: later key");
    let view = view_of(&root, &[("a", "bharathvbcr/a")]);
    let (rows, report) = commit_type::candidates(&view, 7, &no_blobs()).unwrap();
    assert_eq!(report["per_repo"]["a"]["dropped"]["secret_like"], 2, "{report}");
    assert_eq!(rows.len(), 1);
    assert!(rows.iter().all(|c| qd_runtime::caller_record::secret_like_prefix(&c.context).is_none()));
}

#[test]
fn gitpulse_is_refused_by_case_by_directory_by_origin_and_a_substituted_slug() {
    let root = scratch("ct-producer");
    let g = mk_repo_at(&root, "devtools/GitPulse", "bharathvbcr/gitpulse");
    commit_file(&g, "x.txt", b"x\n", "feat: x");
    // A case variant of the slug.
    let v = view_of(&root, &[("devtools/GitPulse", "BharathVBCR/GitPulse")]);
    assert!(commit_type::candidates(&v, 7, &no_blobs()).unwrap_err().contains("refused as a producer"));
    // Another slug on GitPulse's directory.
    let v = view_of(&root, &[("devtools/GitPulse", "bharathvbcr/academiatrack")]);
    assert!(commit_type::candidates(&v, 7, &no_blobs()).unwrap_err().contains("refused as a producer"));
    // GitPulse's content under a renamed directory and an honest-looking slug: its origin says so.
    let r = mk_repo_at(&root, "apps/Renamed", "bharathvbcr/gitpulse");
    commit_file(&r, "x.txt", b"x\n", "feat: x");
    let v = view_of(&root, &[("apps/Renamed", "bharathvbcr/renamed")]);
    assert!(commit_type::candidates(&v, 7, &no_blobs()).unwrap_err().contains("origin remote"));
    // A slug that is not the repository's origin is refused too.
    let m = mk_repo_at(&root, "apps/Mismatch", "bharathvbcr/other");
    commit_file(&m, "x.txt", b"x\n", "feat: x");
    let v = view_of(&root, &[("apps/Mismatch", "bharathvbcr/mismatch")]);
    assert!(commit_type::candidates(&v, 7, &no_blobs()).unwrap_err().contains("not the repository's origin"));
}

#[test]
fn queries_under_the_apps_routing_floor_are_refused_and_counted() {
    let d = scratch("floor");
    let path = route_config(
        &d,
        &phrasings_with(vec![
            ("floor-1", "none_direct", "hi"),
            ("floor-2", "none_direct", "=what is {country}"),
            ("floor-3", "none_direct", ">what is {country}"),
            ("floor-4", "none_direct", "okay"),
        ]),
        None,
    );
    let (cfg, _) = Config::load(&path).unwrap();
    let g = synth::generate(&cfg).unwrap();
    // Three rows per template: "hi", "=...", ">..." are never sent by the app; "okay" is 4 chars.
    assert_eq!(g.report["refused"]["under_app_floor"], 9, "{}", g.report);
    assert!(g.drafts.iter().all(|x| x.context.trim().chars().count() >= 4
        && !x.context.starts_with(['=', '>'])));
    assert!(g.drafts.iter().any(|x| x.template == "floor-4"));
}

#[test]
fn a_route_build_without_the_config_key_or_with_a_missing_ruling_refuses() {
    let d = scratch("report-only-keys");
    let cfg_path = route_config(&d, &phrasings_with(vec![]), None);
    let heldout = d.join("heldout").join("route");
    fs::create_dir_all(heldout.parent().unwrap()).unwrap();
    synth::emit_heldout(&cfg_path, &heldout).unwrap();
    let target = heldout.join("targets.jsonl");
    let sha = qd_prep::sha256::sha256_hex(&fs::read(&target).unwrap());
    let mut cfg: Value = serde_json::from_slice(&fs::read(&cfg_path).unwrap()).unwrap();
    cfg["target_sha256"] = json!({"route-heldout": sha});
    let run = |cfg: &Value, tag: &str| {
        let p = d.join(format!("config-{tag}.json"));
        fs::write(&p, serde_json::to_vec(cfg).unwrap()).unwrap();
        let inputs = synth::Inputs {
            config: p,
            targets: vec![("route-heldout".to_owned(), target.clone())],
        };
        synth::run(&inputs, &d.join(format!("pool-{tag}")), 2)
    };
    // The key present builds (the committed config carries it).
    run(&cfg, "ok").expect("both keys: the build runs");
    // The key absent: refused, naming the missing key.
    let mut no_key = cfg.clone();
    no_key["probe"].as_object_mut().unwrap().remove("report_only");
    let e = run(&no_key, "nokey").unwrap_err();
    assert!(e.contains("probe.report_only.devtype.route.ruling") && e.contains("both keys or neither"), "{e}");
    // The key naming a ruling that is not the one the code names.
    let mut wrong = cfg.clone();
    wrong["probe"]["report_only"]["devtype.route"]["ruling"] = json!("AUDIT/not-a-ruling.md");
    assert!(run(&wrong, "wrong").unwrap_err().contains("not the ruling the code names"));
}
