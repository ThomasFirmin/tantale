pub mod sol;
pub use sol::Solution;

pub mod partial;
pub use partial::{Partial,PartialSol};

pub mod computed;
pub use computed::{Computed,ComputedSol};

/// [`ArcSol`]  is a type alias for an [`Arc`] of a [`Solution`]. Used at the output of an [`Optimizer`] and
/// for multi-threading, in [`run`](tantale::core::experiment::Experiment::run) of an [`Experiment`](tantale::core::Experiment).
pub type ArcSol<Dom, Cod, Out, Info, const N: usize> = std::sync::Arc<ComputedSol<Dom, Cod, Out, Info,N>>;

/// Slice of [`ArcSol`].
pub type VecArcSol<Dom, Cod, Out, Info, const N:usize> = std::sync::Arc<Vec<ArcSol<Dom, Cod, Out, Info,N>>>;