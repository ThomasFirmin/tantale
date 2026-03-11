use crate::{
    objective::Step,
    stop::{ExpStep, Stop},
};
use serde::{Deserialize, Serialize};

/// A [`Stop`] criterion, returning `false` when the number of
/// [`Evaluated`](crate::objective::Step::Evaluated) solutions during a run exceed a threshold.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Evaluated(pub usize, pub usize);

impl Stop for Evaluated {
    fn init(&mut self) {} // No initialization needed

    fn stop(&self) -> bool {
        self.0 >= self.1
    }

    /// Update the number of calls with a new [`ExpStep`], incrementing the counter if the step
    /// is a fully [`Evaluated`](crate::objective::Step::Evaluated) evaluation. Other steps do not update the counter.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{objective::Step, stop::{Stop, Evaluated, ExpStep}};
    /// let mut calls = Evaluated::new(3);
    /// calls.update(ExpStep::Distribution(Step::Evaluated));
    /// calls.update(ExpStep::Distribution(Step::Partially(5)));
    /// calls.update(ExpStep::Distribution(Step::Discard));
    /// calls.update(ExpStep::Distribution(Step::Error));
    /// assert_eq!(calls.calls(), 1);
    fn update(&mut self, step: ExpStep) {
        if let ExpStep::Distribution(Step::Evaluated) = step {
            self.0 += 1
        }
    }
}

impl Evaluated {
    /// Create a new [`Evaluated`] stop criterion with a given threshold.
    pub fn new(threshold: usize) -> Self {
        Evaluated(0, threshold)
    }
    /// Return the number of [`Evaluated`](crate::objective::Step::Evaluated) solutions so far.
    pub fn calls(&self) -> usize {
        self.0
    }

    /// Increase the call counter by a given amount.
    /// This can be used to extend the call limit of a run.
    pub fn add(&mut self, count: usize) {
        self.1 += count;
    }
}
