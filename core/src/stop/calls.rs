use crate::{
    objective::Step,
    stop::{ExpStep, Stop}
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
        if let ExpStep::Distribution(Step::Evaluated) =  step{
            self.0 += 1
        }
    }
}

impl Calls {
    pub fn new(max: usize) -> Self {
        Calls(0, max)
    }
    pub fn calls(&self) -> usize {
        self.0
    }
}
