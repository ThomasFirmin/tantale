pub enum ExpStep {
    Evaluation,
    Distribution,
    Optimization,
    Never,
}

pub trait Stop {
    fn init(&mut self);
    fn stop(&self) -> bool;
    fn update(&mut self, step: ExpStep);
}

pub mod calls;
pub use calls::Calls;
