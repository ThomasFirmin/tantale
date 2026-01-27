use crate::{
    objective::Step,
    stop::{ExpStep, Stop},
};
use serde::{Deserialize, Serialize};

/// A [`Stop`] criterion, returning false when the number of
/// evaluation during a run exceed a threshold.
#[derive(std::fmt::Debug, Serialize, Deserialize)]
pub struct Calls(pub usize, pub usize);

impl Stop for Calls {
    fn init(&mut self) {
        self.1 = 0;
    }

    fn stop(&self) -> bool {
        self.0 >= self.1
    }

    fn update(&mut self, step: ExpStep) {
        match step {
            ExpStep::Distribution(Step::Evaluated) => self.0 += 1,
            ExpStep::Distribution(Step::Discard) => self.0 += 1,
            _ => {}
        }
    }
}

impl Calls {
    pub fn new(threshold: usize) -> Self {
        Calls(0, threshold)
    }
    pub fn calls(&self) -> usize {
        self.0
    }
}
