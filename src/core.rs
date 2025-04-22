//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

pub mod domain;
pub use domain::Domain;
pub use domain::{Bounded,DomainBounded,Real,Nat,Int};
pub use domain::Bool;
pub use domain::Cat;
pub use domain::Unit;
pub use domain::{BaseDom,BaseTypeDom};
pub use domain::Onto;
pub use domain::{uniform,uniform_real,uniform_nat,uniform_int,uniform_bool,uniform_cat};
pub use domain::{DomainError, DomainBoundariesError,DomainOoBError};


pub mod errors;
pub mod objective;
pub use crate::core::objective::Objective;
pub mod optimizer;
pub use crate::core::optimizer::Optimizer;
pub mod variable;
// pub use crate::core::variable::Variable;
pub mod element;
pub mod solution;
// pub use crate::core::solution::Solution;
pub mod searchspace;