//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

pub mod domain;
pub use crate::core::domain::{Bool, Bounded, Cat, Domain, DomainBounded, Int, Nat, Real, Unit};
pub mod onto;

pub mod errors;
pub mod objective;
pub use crate::core::objective::Objective;
pub mod optimizer;
pub use crate::core::optimizer::Optimizer;
pub mod sampler;
pub mod variable;
pub use crate::core::variable::Variable;
pub mod element;
pub use crate::core::element::Element;
pub mod solution;
pub use crate::core::solution::Solution;
