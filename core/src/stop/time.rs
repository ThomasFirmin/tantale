use crate::stop::{ExpStep, Stop};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// A [`Stop`] criterion, returning `false` when the elapsed time during a run exceeds a threshold duration.
///
/// This criterion tracks elapsed time across checkpoints. When resumed from a checkpoint,
/// it preserves the accumulated time from before the checkpoint and continues counting
/// from the remaining duration.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Time(Duration, Duration, #[serde(skip)] Option<Instant>); // (threshold, accumulated, start time)

impl Stop for Time {
    fn init(&mut self) {
        self.2 = Some(Instant::now());
    }

    fn stop(&self) -> bool {
        let current_elapsed = self.2.map(|start| start.elapsed()).unwrap_or_default();
        self.1 + current_elapsed <= self.0
    }

    /// Update the time criterion. This does not increment any counter but allows
    /// the criterion to track elapsed time since [`init`](Self::init) was called.
    /// Other [`ExpStep`] variants are ignored.
    ///
    /// # Example
    /// ```
    /// use tantale::core::stop::{Time, ExpStep};
    /// use std::time::Duration;
    /// use std::thread;
    ///
    /// let mut time = Time::new(Duration::from_millis(100));
    /// time.init();
    /// time.update(ExpStep::Iteration);
    /// assert!(!time.stop()); // Still within time limit
    /// thread::sleep(Duration::from_millis(150));
    /// assert!(time.stop()); // Time limit exceeded
    fn update(&mut self, _step: ExpStep) {
        self.1 += self.2.map(|start| start.elapsed()).unwrap_or_default();
        self.2 = Some(Instant::now());
    }
}

impl Time {
    /// Create a new [`Time`] stop criterion with a given duration threshold.
    pub fn new(duration: Duration) -> Self {
        Time(duration, Duration::ZERO, None)
    }

    /// Return the total elapsed duration including accumulated time from previous checkpoints.
    ///
    /// Returns `None` if [`init`](Self::init) has not been called yet.
    pub fn elapsed(&self) -> Option<Duration> {
        self.2.map(|start| self.1 + start.elapsed())
    }

    /// Increase the duration threshold by a given amount.
    /// This can be used to extend the time limit of a run.
    pub fn add(&mut self, duration: Duration) {
        self.0 += duration;
    }
}