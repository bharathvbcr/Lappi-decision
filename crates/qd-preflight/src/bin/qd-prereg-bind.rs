//! `qd-prereg-bind`: refuse a run that is not bound to its committed pre-registration. Exit 0
//! PASS, 1 FAILED, 3 NOT_RUN (a check could not run), 2 a bad command line. The last stdout line
//! is `RESULT: PASS|FAILED|NOT_RUN`.
//!
//! `qd-prereg-bind --prereg campaign/lora-2b-preregistered.json --shard-dir DIR [--repo .] [--head SHA]`

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (report, code) = qd_preflight::prereg_bind::run(&args);
    if code == 2 {
        eprint!("{report}");
    } else {
        print!("{report}");
    }
    ExitCode::from(code)
}
