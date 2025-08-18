//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

pub mod domain;
pub use domain::{
    uniform, uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real, BaseDom,
    BaseTypeDom, Bool, Bounded, Cat, Domain, DomainBoundariesError, DomainBounded, DomainError,
    DomainOoBError, Int, Nat, Onto, Real, Unit,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{Computed, Id, ParSId, Partial, PartialSol, SId, SolInfo, Solution};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Criteria, FidelCodomain,
    FidelConstCodomain, FidelConstMultiCodomain, FidelMultiCodomain, Fidelity, HashOut,
    LinkedOutcome, Multi, MultiCodomain, ObjBase, Objective, Outcome, Single, SingleCodomain,
};

pub mod optimizer;
pub use crate::optimizer::{ArcVecArc, EmptyInfo, OptInfo, Optimizer};

pub mod stop;
pub use stop::Stop;

pub mod experiment;

pub mod saver;
pub use saver::CSVSaver;
