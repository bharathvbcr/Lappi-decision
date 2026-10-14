//! `qd-pin`: the sibling-pin gate. Exit 0 the sibling is at the locked commit, 1 it is at another
//! commit (both are printed), 3 the check could not run (sibling or lock unreadable), 2 a bad
//! command line.
//!
//! `qd-pin --lock upstream.lock.json --sibling ../ojas [--key ojas]`

use std::path::PathBuf;
use std::process::ExitCode;

use qd_preflight::pin::check_pin;

fn main() -> ExitCode {
    let mut lock: Option<PathBuf> = None;
    let mut sibling: Option<PathBuf> = None;
    let mut key = "ojas".to_owned();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            return usage(&format!("{flag} needs a value"));
        };
        match flag.as_str() {
            "--lock" => lock = Some(PathBuf::from(value)),
            "--sibling" => sibling = Some(PathBuf::from(value)),
            "--key" => key = value,
            other => return usage(&format!("unknown argument {other:?}")),
        }
    }
    let (Some(lock), Some(sibling)) = (lock, sibling) else {
        return usage("--lock and --sibling are required");
    };
    let outcome = check_pin(&lock, &key, &sibling);
    println!("{}", outcome.line(&key));
    ExitCode::from(outcome.exit_code())
}

fn usage(why: &str) -> ExitCode {
    eprintln!("qd-pin: {why}\nusage: qd-pin --lock FILE --sibling DIR [--key ojas]");
    ExitCode::from(2)
}
