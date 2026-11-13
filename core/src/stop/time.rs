use crate::{
    HasState, State,
    stop::{ExpStep, Stop},
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// A [`Stop`] criterion that becomes true when the elapsed time reaches its limit.
///
/// This criterion tracks elapsed time across checkpoints. When resumed from a checkpoint,
/// it preserves the accumulated time from before the checkpoint and continues counting
/// from the remaining duration.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Time(Duration, Duration, #[serde(skip)] Option<Instant>); // (threshold, accumulated, start time)

impl State for Time {}
impl HasState for Time {
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl Stop for Time {
    fn init(&mut self) {
        self.2 = Some(Instant::now());
    }

    fn stop(&self) -> bool {
        let current_elapsed = self.2.map(|start| start.elapsed()).unwrap_or_default();
        self.1 + current_elapsed >= self.0
    }

    /// Update the time criterion. This does not increment any counter but allows
    /// the criterion to track elapsed time since [`init`](Self::init) was called.
    /// Other [`ExpStep`] variants are ignored.
    ///
    /// # Example
    /// ```
    /// use tantale::core::stop::{Stop, Time, ExpStep};
    /// use std::time::Duration;
    ///
    /// let mut time = Time::new(Duration::from_secs(60));
    /// time.init();
    /// time.update(ExpStep::Iteration);
    /// assert!(!time.stop()); // The limit has not been reached.
    ///
    /// let mut immediate = Time::new(Duration::ZERO);
    /// immediate.init();
    /// assert!(immediate.stop()); // The zero-duration limit is reached immediately.
    /// ```
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_is_false_before_the_limit() {
        let mut time = Time::new(Duration::from_secs(60));
        time.init();
        assert!(!time.stop());
    }

    #[test]
    fn stop_is_true_when_elapsed_time_reaches_the_limit() {
        let time = Time(Duration::from_secs(1), Duration::from_secs(1), None);
        assert!(time.stop());
    }
}
