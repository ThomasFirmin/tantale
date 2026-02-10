use crate::objective::Step;
use serde::{Deserialize, Serialize};

pub enum ExpStep {
    Distribution(Step),
    Iteration,
    Optimization,
    Never,
}

pub trait Stop
where
    Self: Serialize + for<'a> Deserialize<'a>,
{
    fn init(&mut self);
    fn stop(&self) -> bool;
    fn update(&mut self, step: ExpStep);
}

pub mod calls;
pub use calls::Calls;

pub mod time;
pub use time::Time;