//! A team of threads started once and reused for every phase of a loop.
//!
//! The FT linear control's fit runs two parallel phases an iteration, and up to ~30,000
//! iterations a task. Starting the threads for each phase cost ~32 us a thread a phase on the
//! v5 H100 box (2026-10-04, `examples/linfit_bench.rs`, CPUs 26-51 at nice 19): at 52 threads
//! ~3.4 ms an iteration, which was the whole difference between the box's 5.4 ms an iteration
//! on openjev.game and the arithmetic's share of it. A team pays that once per fit.
//!
//! [`with_team`] starts `threads - 1` workers inside a [`std::thread::scope`]; the calling
//! thread leads. The phases' work is fixed when the team starts: `jobs[j](i)` is item `i` of a
//! phase of job `j`. [`Team::run`] publishes one phase, runs the leader's own `meanwhile` work,
//! then takes items like any worker, and returns once every item has run. Items are claimed
//! one at a time from a counter stamped with the phase, so an item runs exactly once, a worker
//! that wakes late finds the phase taken and waits for the next, and a phase never waits for a
//! thread that holds no item -- a worker starved by the scheduler (the box's lanes run at a
//! higher priority than a control) delays only an item it already holds.
//!
//! The jobs share the loop's state through [`SharedF64s`], whose reads and writes are relaxed
//! atomics: every item writes elements no other item of its phase touches, and each phase is
//! ordered after the last by the phase counter (released by the leader when it publishes,
//! acquired by every claim) and by the completion count (released by every item, acquired by
//! the leader before it returns). So which thread runs which item cannot change any value, and
//! the crate stays free of `unsafe`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// One phase's work: item `i`.
pub type Job<'j> = &'j (dyn Fn(usize) + Sync);

/// How long a thread polls for the next phase, or the leader for the last item, before it
/// sleeps. The leader's work between two phases is short (a few reductions), so a worker that
/// polls this long usually catches the next phase without a wake-up.
const SPIN: Duration = Duration::from_micros(50);
/// A phase that has not finished in this long is a hang, not a slow fit: the largest v5 task's
/// whole iteration took ~0.25 s on the box.
const PHASE_LIMIT: Duration = Duration::from_secs(3600);

/// `f64` cells that the threads of a team read and write between phases.
pub struct SharedF64s(Box<[AtomicU64]>);

impl SharedF64s {
    pub fn zeros(n: usize) -> Self {
        Self((0..n).map(|_| AtomicU64::new(0f64.to_bits())).collect())
    }

    pub fn from_slice(values: &[f64]) -> Self {
        Self(values.iter().map(|v| AtomicU64::new(v.to_bits())).collect())
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> f64 {
        f64::from_bits(self.0[i].load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn set(&self, i: usize, value: f64) {
        self.0[i].store(value.to_bits(), Ordering::Relaxed);
    }

    /// Cells `at..at + len`, bounds-checked once for a loop that reads them all with [`load`].
    #[inline(always)]
    pub fn row(&self, at: usize, len: usize) -> &[AtomicU64] {
        &self.0[at..at + len]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn to_vec(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }
}

/// The `f64` in one cell of a [`SharedF64s::row`].
#[inline(always)]
pub fn load(cell: &AtomicU64) -> f64 {
    f64::from_bits(cell.load(Ordering::Relaxed))
}

/// The phase being run, as the workers read it.
struct Phase {
    epoch: u32,
    job: usize,
    n_items: usize,
    stop: bool,
}

struct Shared {
    phase: Mutex<Phase>,
    published: Condvar,
    /// `epoch << 32 | next item`: a claim succeeds only for the phase it read.
    ticket: AtomicU64,
    done: AtomicUsize,
    finished: Mutex<()>,
    all_done: Condvar,
    panicked: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    /// Claim and run items of phase `epoch` of `job` until none is left.
    fn work(&self, job: Job<'_>, epoch: u32, n_items: usize) {
        loop {
            let t = self.ticket.load(Ordering::Acquire);
            if (t >> 32) as u32 != epoch {
                return;
            }
            let item = (t & u64::from(u32::MAX)) as usize;
            if item >= n_items {
                return;
            }
            if self
                .ticket
                .compare_exchange_weak(t, t + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // A panicking item still counts as done, so the leader is never left waiting on
            // it; the leader then panics in its place once the phase is over.
            if catch_unwind(AssertUnwindSafe(|| job(item))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == n_items {
                let _held = lock(&self.finished);
                self.all_done.notify_all();
            }
        }
    }

    fn worker(&self, jobs: &[Job<'_>]) {
        let mut seen = 0u32;
        loop {
            let polled = Instant::now();
            while (self.ticket.load(Ordering::Acquire) >> 32) as u32 == seen
                && polled.elapsed() < SPIN
            {
                std::hint::spin_loop();
            }
            let (epoch, job, n_items) = {
                let mut p = lock(&self.phase);
                while p.epoch == seen && !p.stop {
                    p = self
                        .published
                        .wait(p)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                if p.stop {
                    return;
                }
                (p.epoch, p.job, p.n_items)
            };
            seen = epoch;
            self.work(jobs[job], epoch, n_items);
        }
    }
}

/// The leader's handle on a running team.
pub struct Team<'t> {
    shared: &'t Shared,
    jobs: &'t [Job<'t>],
    epoch: u32,
}

impl Team<'_> {
    /// Run every item `0..n_items` of `jobs[job]` on the team, while this thread first runs
    /// `meanwhile` and then takes items too; returns `meanwhile`'s value once every item has
    /// run. Panics if an item panicked, or if the phase has not finished within an hour.
    pub fn run<R>(&mut self, job: usize, n_items: usize, meanwhile: impl FnOnce() -> R) -> R {
        assert!(job < self.jobs.len(), "job {job} of {}", self.jobs.len());
        assert!(
            n_items < u32::MAX as usize,
            "{n_items} items is past the ticket's range"
        );
        let shared = self.shared;
        // Epoch 0 is the state before the first phase, which no claim may match.
        self.epoch = self.epoch.wrapping_add(1).max(1);
        let epoch = self.epoch;
        shared.done.store(0, Ordering::Relaxed);
        {
            let mut p = lock(&shared.phase);
            p.epoch = epoch;
            p.job = job;
            p.n_items = n_items;
            shared
                .ticket
                .store(u64::from(epoch) << 32, Ordering::Release);
        }
        shared.published.notify_all();

        let out = meanwhile();
        shared.work(self.jobs[job], epoch, n_items);

        let started = Instant::now();
        while shared.done.load(Ordering::Acquire) != n_items {
            if started.elapsed() < SPIN {
                std::hint::spin_loop();
                continue;
            }
            assert!(
                started.elapsed() < PHASE_LIMIT,
                "a phase of job {job} has not finished in {PHASE_LIMIT:?}: {} of {n_items} items",
                shared.done.load(Ordering::Acquire)
            );
            let held = lock(&shared.finished);
            if shared.done.load(Ordering::Acquire) == n_items {
                break;
            }
            drop(
                shared
                    .all_done
                    .wait_timeout(held, Duration::from_millis(10))
                    .unwrap_or_else(PoisonError::into_inner),
            );
        }
        assert!(
            !shared.panicked.load(Ordering::Relaxed),
            "an item of job {job} panicked"
        );
        out
    }
}

/// Tells the workers to stop when the leader leaves, by return or by panic: the scope joins
/// them before it returns, so a worker left waiting for a phase would hang the scope.
struct StopOnDrop<'s>(&'s Shared);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        lock(&self.0.phase).stop = true;
        self.0.published.notify_all();
    }
}

/// Start `threads - 1` workers for `jobs` and run `body` as their leader. `threads` of 0 or 1
/// starts none: the leader runs every item itself.
pub fn with_team<R>(threads: usize, jobs: &[Job<'_>], body: impl FnOnce(&mut Team<'_>) -> R) -> R {
    let shared = Shared {
        phase: Mutex::new(Phase {
            epoch: 0,
            job: 0,
            n_items: 0,
            stop: false,
        }),
        published: Condvar::new(),
        ticket: AtomicU64::new(0),
        done: AtomicUsize::new(0),
        finished: Mutex::new(()),
        all_done: Condvar::new(),
        panicked: AtomicBool::new(false),
    };
    std::thread::scope(|scope| {
        let shared = &shared;
        for _ in 1..threads.max(1) {
            scope.spawn(move || shared.worker(jobs));
        }
        #[cfg(test)]
        STARTED.with(|s| s.set(s.get() + threads.max(1) - 1));
        let _stop = StopOnDrop(shared);
        let mut team = Team {
            shared,
            jobs,
            epoch: 0,
        };
        body(&mut team)
    })
}

#[cfg(test)]
thread_local! {
    /// Threads started from this thread by [`with_team`], for the tests that count them.
    pub static STARTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_of_every_phase_runs_exactly_once() {
        for threads in [0usize, 1, 2, 3, 8, 33] {
            let counts: Vec<SharedF64s> = (0..2).map(|_| SharedF64s::zeros(97)).collect();
            let bump = |j: usize| {
                let counts = &counts[j];
                move |i: usize| {
                    // One writer per item, so a plain read-then-write is exact here.
                    counts.set(i, counts.get(i) + 1.0);
                }
            };
            let (a, b) = (bump(0), bump(1));
            let jobs: [Job<'_>; 2] = [&a, &b];
            with_team(threads, &jobs, |team| {
                for round in 0..200 {
                    team.run(0, 97, || ());
                    // Phases of different sizes, including empty and single-item ones.
                    team.run(1, round % 98, || ());
                }
            });
            for i in 0..97 {
                assert_eq!(counts[0].get(i), 200.0, "{threads} threads, job 0 item {i}");
                // Item i of job 1 ran in every round r with r % 98 > i.
                let expected = (0..200usize).filter(|r| r % 98 > i).count() as f64;
                assert_eq!(
                    counts[1].get(i),
                    expected,
                    "{threads} threads, job 1 item {i}"
                );
            }
        }
    }

    #[test]
    fn a_phase_sees_the_last_phase_and_the_leaders_writes() {
        // Phase 0 doubles every cell into `next`; the leader then copies `next` back between
        // phases. Any lost ordering shows as a wrong power of two.
        let cells = SharedF64s::from_slice(&[1.0; 64]);
        let next = SharedF64s::zeros(64);
        let double = |i: usize| next.set(i, cells.get(i) * 2.0);
        let jobs: [Job<'_>; 1] = [&double];
        with_team(7, &jobs, |team| {
            for _ in 0..40 {
                team.run(0, 64, || ());
                for i in 0..64 {
                    cells.set(i, next.get(i));
                }
            }
        });
        assert!(cells.to_vec().iter().all(|&v| v == 2f64.powi(40)));
    }

    #[test]
    fn meanwhile_runs_on_the_leader_and_its_value_is_returned() {
        let noop = |_: usize| {};
        let jobs: [Job<'_>; 1] = [&noop];
        let leader = std::thread::current().id();
        let got = with_team(4, &jobs, |team| {
            team.run(0, 10, || std::thread::current().id())
        });
        assert_eq!(got, leader);
    }

    #[test]
    fn workers_start_once_per_team_not_per_phase() {
        let noop = |_: usize| {};
        let jobs: [Job<'_>; 1] = [&noop];
        let before = STARTED.with(std::cell::Cell::get);
        with_team(5, &jobs, |team| {
            for _ in 0..100 {
                team.run(0, 16, || ());
            }
        });
        assert_eq!(STARTED.with(std::cell::Cell::get) - before, 4);
    }

    #[test]
    #[should_panic(expected = "an item of job 0 panicked")]
    fn a_panicking_item_panics_the_leader_instead_of_hanging_it() {
        let boom = |i: usize| assert!(i != 5, "item 5 refuses");
        let jobs: [Job<'_>; 1] = [&boom];
        with_team(4, &jobs, |team| {
            team.run(0, 10, || ());
        });
    }
}
