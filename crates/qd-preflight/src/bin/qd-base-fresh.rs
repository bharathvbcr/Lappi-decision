//! `qd-base-fresh`: refuse a checkout built on a stale base. Exit 0 the target branch is an
//! ancestor of HEAD, 1 the branch has commits HEAD lacks (the count is printed), 3 git could not
//! answer, 2 a bad command line. The last stdout line is `RESULT: PASS|FAILED|NOT_RUN`.
//!
//! `qd-base-fresh --repo PATH [--into main]`

use std::path::PathBuf;
use std::process::ExitCode;

use qd_preflight::base_fresh::check_base;

fn main() -> ExitCode {
    let mut repo: Option<PathBuf> = None;
    let mut into = "main".to_owned();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            return usage(&format!("{flag} needs a value"));
        };
        match flag.as_str() {
            "--repo" => repo = Some(PathBuf::from(value)),
            "--into" => into = value,
            other => return usage(&format!("unknown argument {other:?}")),
        }
    }
    let Some(repo) = repo else {
        return usage("--repo is required");
    };
    let outcome = check_base(&repo, &into);
    println!("{}", outcome.line());
    println!("{}", outcome.result());
    ExitCode::from(outcome.exit_code())
}

fn usage(why: &str) -> ExitCode {
    eprintln!("qd-base-fresh: {why}\nusage: qd-base-fresh --repo PATH [--into main]");
    ExitCode::from(2)
}
