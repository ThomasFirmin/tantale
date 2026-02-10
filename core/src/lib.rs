//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

use serde::{Deserialize, Serialize};
#[cfg(feature = "mpi")]
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

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
    Mixed, MixedTypeDom, Bool, Bounded, Cat, Codomain, ConstCodomain, ConstMultiCodomain,
    Constrained, Cost, CostCodomain, CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain,
    Criteria, Domain, FidCriteria, HasEvalStep, Int, Multi, MultiCodomain, Nat, Onto, Real, Single,
    SingleCodomain, Unit,
};

pub mod sampler;
pub use sampler::{Bernoulli,Uniform,Sampler,BoundedDistribution,CatDistribution,BoolDistribution};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{
    BasePartial, Computed, FidBasePartial, Fidelity, Id, ParSId, SId, SolInfo, Solution, HasFidelity,HasId,HasInfo,HasSolInfo,HasStep,HasUncomputed,HasY,
    Batch, OutBatch
};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp, SpPar};

pub mod errors;

pub mod objective;
pub use crate::objective::{EvalStep, FidOutcome, FuncWrapper, Objective, Outcome, Stepped, outcome::FuncState};

pub mod optimizer;
pub use crate::optimizer::{EmptyInfo, OptInfo, Optimizer,BatchOptimizer,SequentialOptimizer, OptState};

pub mod stop;
pub use stop::{Stop,Calls};

pub mod experiment;
pub use experiment::{Runable,MonoExperiment,ThrExperiment,mono,threaded};
#[cfg(feature = "mpi")]
pub use experiment::{MPIExperiment, DistEvaluate,MasterWorker, distributed, mpi::utils::MPIProcess, mpi::worker::Worker};

pub mod recorder;
pub use recorder::{Recorder,CSVRecorder};
#[cfg(feature = "mpi")]
pub use recorder::{DistRecorder};

pub mod checkpointer;
pub use checkpointer::{Checkpointer,ThrCheckpointer,MessagePack};
#[cfg(feature = "mpi")]
pub use checkpointer::{DistCheckpointer};