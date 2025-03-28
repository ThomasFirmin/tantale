//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

pub mod domain;
pub use crate::core::domain::{Bool, Cat, Domain, Int, Nat, NumericallyBounded, Real};
pub mod convertible;

pub mod errors;
pub mod objective;
pub mod optimizer;
pub mod sampler;
pub mod variable;
pub use crate::core::variable::{DomainObjective, DomainOptimizer, Variable};

pub mod check;
