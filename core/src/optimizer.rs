pub mod opt;
pub use opt::{ArcVecArc, EmptyInfo, OptInfo, OptState, Optimizer};

pub enum Parallel {
    Sequential,
    MultiThread,
    MultiProcess,
    Distributed,
}