use crate::{
    HasState, State,
    objective::Step,
    stop::{ExpStep, Stop},
};
use serde::{Deserialize, Serialize};

/// A [`Stop`] criterion that terminates after the configured number of calls to the objective function.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Calls(
    /// Represents the number of calls made so far.
    pub usize, 
    /// Represents the threshold for stopping.
    pub usize
);

impl State for Calls {}
impl HasState for Calls {
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

impl Stop for Calls {
    fn init(&mut self) {} // No initialization needed

    fn stop(&self) -> bool {
        self.0 >= self.1
    }

    /// Update the number of calls to the objective function with a new [`ExpStep`], incrementing the counter if the step is 
    /// a fully [`Evaluated`](crate::objective::Step::Evaluated) evaluation or a [`Partially`](crate::objective::Step::Partially) evaluated step. 
    /// Other steps do not update the counter, as they do not represent a call to the objective function.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Step, stop::{Stop, Calls, ExpStep}};
    /// let mut calls = Calls::new(3);
    /// calls.update(ExpStep::Evaluation(Step::Evaluated));
    /// calls.update(ExpStep::Evaluation(Step::Partially(5)));
    /// calls.update(ExpStep::Evaluation(Step::Discard));
    /// assert_eq!(calls.calls(), 2);
    fn update(&mut self, step: ExpStep) {
        match step {
            ExpStep::Evaluation(Step::Evaluated) | ExpStep::Evaluation(Step::Partially(_))  => self.0 += 1,
            _ => {}
        }
    }
}

impl Calls {
    /// Create a new [`Calls`] stop criterion with a given threshold.
    pub fn new(threshold: usize) -> Self {
        Calls(0, threshold)
    }
    /// Return the number of calls to the objective function that have been made so far.
    pub fn calls(&self) -> usize {
        self.0
    }

    /// Increase the call counter by a given amount.
    /// This can be used to extend the call limit of a run.
    pub fn add(&mut self, count: usize) {
        self.1 += count;
    }
}
