//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

pub mod domain;
pub use domain::Bool;
pub use domain::Cat;
pub use domain::Domain;
pub use domain::Onto;
pub use domain::Unit;
pub use domain::{uniform, uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real};
pub use domain::{BaseDom, BaseTypeDom};
pub use domain::{Bounded, DomainBounded, Int, Nat, Real};
pub use domain::{DomainBoundariesError, DomainError, DomainOoBError};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::Solution;

pub mod searchspace;
pub use searchspace::{Sp,Searchspace};
#[cfg(feature="par")]
pub use searchspace::ParSearchspace;

pub mod errors;

pub mod objective;
pub use crate::objective::{Objective, SimpleObjective,Criteria, Codomain, Single, Multi, Constrained, Fidelity, Outcome, HashOut};

pub mod optimizer;
pub use crate::optimizer::Optimizer;

pub mod stop;
pub use stop::Stop;

