use crate::{
    objective::Step,
    stop::{ExpStep, Stop},
};
use serde::{Deserialize, Serialize};

/// A [`Stop`] criterion, returning `false` when the number of
/// [`Evaluated`](crate::objective::Step::Evaluated) or [`Discard`](crate::objective::Step::Discard) solutions during a run exceed a threshold.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Calls(pub usize, pub usize);

impl Stop for Calls {
    fn init(&mut self) {} // No initialization needed

    fn stop(&self) -> bool {
        self.0 >= self.1
    }

    /// Update the number of calls with a new [`ExpStep`], incrementing the counter if the step is a fully [`Evaluated`](crate::objective::Step::Evaluated)
    /// or [`Discard`](crate::objective::Step::Discard) evaluation. Other steps do not update the counter.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Step, stop::{Stop, Calls, ExpStep}};
    /// let mut calls = Calls::new(3);
    /// calls.update(ExpStep::Distribution(Step::Evaluated));
    /// calls.update(ExpStep::Distribution(Step::Partially(5)));
    /// calls.update(ExpStep::Distribution(Step::Discard));
    /// calls.update(ExpStep::Distribution(Step::Error));
    /// assert_eq!(calls.calls(), 2);
    fn update(&mut self, step: ExpStep) {
        match step {
            ExpStep::Distribution(Step::Evaluated) => self.0 += 1,
            ExpStep::Distribution(Step::Discard) => self.0 += 1,
            _ => {}
        }
    }
}

impl Calls {
    /// Create a new [`Calls`] stop criterion with a given threshold.
    pub fn new(threshold: usize) -> Self {
        Calls(0, threshold)
    }
    /// Return the number of [`Evaluated`](crate::objective::Step::Evaluated) and [`Discard`](crate::objective::Step::Discard) solutions so far.
    pub fn calls(&self) -> usize {
        self.0
    }

    /// Increase the call counter by a given amount.
    /// This can be used to extend the call limit of a run.
    pub fn add(&mut self, count: usize) {
        self.1 += count;
    }
}
