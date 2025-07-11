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
pub use solution::{Solution,PartialSol,ComputedSol};

pub mod searchspace;
#[cfg(feature = "par")]
pub use searchspace::ParSearchspace;
pub use searchspace::{Searchspace, Sp};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Criteria, FidelCodomain,
    FidelConstCodomain, FidelConstMultiCodomain, FidelMultiCodomain, Fidelity, HashOut, Multi,
    MultiCodomain, Objective, Outcome, ObjBase, Single, SingleCodomain,
};

pub mod optimizer;
pub use crate::optimizer::{EmptyInfo, OptInfo, Optimizer, SolInfo};

pub mod stop;
pub use stop::Stop;

pub mod experiment;
pub use experiment::run;

pub mod saver;