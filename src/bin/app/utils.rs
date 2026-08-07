//! Small helpers for the CLI: timing a call, and turning a panic in the
//! computation into an ordinary error.

use super::app::{AppErr, err};

/// Runs `proc` and returns its result together with the elapsed time.
pub fn measure<F, Res>(proc: F) -> (Res, std::time::Duration)
where F: FnOnce() -> Res {
    let start = std::time::Instant::now();
    let res = proc();
    let time = start.elapsed();
    (res, time)
}

/// Catches a panic from `f` and reports it as an error, so one failed
/// grading does not abort a long batch run.
pub fn guard_panic<F, R>(f: F) -> Result<R, Box<dyn std::error::Error>>
where F: FnOnce() -> Result<R, Box<dyn std::error::Error>> + std::panic::UnwindSafe {
    std::panic::catch_unwind(|| {
        f()
    }).unwrap_or_else(|e| {
        let info = match e.downcast::<String>() {
            Ok(v) => *v,
            Err(e) => match e.downcast::<&str>() {
                Ok(v) => v.to_string(),
                _ => "Unknown Source of Error".to_owned()
            }
        };
        err!("panic: {info}")
    })
}