//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "mpi")]
use std::sync::OnceLock;

pub static SOL_ID: AtomicUsize = AtomicUsize::new(0);
pub static OPT_ID: AtomicUsize = AtomicUsize::new(0);
pub static RUN_ID: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "mpi")]
pub static MPI_RANK: OnceLock<mpi::Rank> = OnceLock::new();
#[cfg(feature = "mpi")]
pub static MPI_SIZE: OnceLock<mpi::Rank> = OnceLock::new();

#[derive(Serialize, Deserialize)]
pub struct GlobalParameters {
    pub sold_id: usize,
    pub opt_id: usize,
    pub run_id: usize,
}

pub mod config;
#[cfg(feature = "mpi")]
pub use config::DistSaverConfig;
pub use config::{FolderConfig, SaverConfig};

pub mod domain;
pub use domain::{
    uniform, uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real, BaseDom,
    BaseTypeDom, Bool, Bounded, Cat, Domain, DomainBoundariesError, DomainBounded, DomainError,
    DomainOoBError, Int, Nat, Onto, Real, Unit,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{BasePartial, Computed, Fidelity, Id, ParSId, Partial, SId, SolInfo, Solution};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, EvalStep, StepCodomain,
    StepConstCodomain, StepConstMultiCodomain, StepCostCodomain, StepCostConstCodomain,
    StepCostConstMultiCodomain, StepCostMultiCodomain, FidCriteria, StepMultiCodomain, FidOutcome,
    Multi, MultiCodomain, Objective, Outcome, Single, SingleCodomain, Stepped,
};

pub mod optimizer;
pub use crate::optimizer::{EmptyInfo, OptInfo, Optimizer};

pub mod stop;
pub use stop::Stop;

pub mod experiment;

pub mod recorder;
pub use recorder::CSVRecorder;

pub mod checkpointer;
pub use checkpointer::MessagePack;
