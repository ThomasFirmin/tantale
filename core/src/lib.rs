//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "mpi")]
use std::sync::OnceLock;
#[cfg(feature="mpi")]
use mpi::Rank;

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

pub trait SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn get_sp(&self)->Arc<Scp>;
    fn get_cod(&self)->Arc<Cod>;
}

#[cfg(feature="mpi")]
pub trait DistSaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>: SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn get_rank(&self)-> Rank;
}

pub struct FolderConfig;

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
#[cfg(feature="mpi")]
use crate::domain::onto::OntoDom;
pub use crate::objective::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost, CostCodomain,
    CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, FidCodomain,
    FidConstCodomain, FidConstMultiCodomain, FidCostCodomain, FidCostConstCodomain,
    FidCostConstMultiCodomain, FidCostMultiCodomain, FidCriteria, FidMultiCodomain, Multi,
    MultiCodomain, Objective, Outcome, Single, SingleCodomain, Stepped, FidOutcome,EvalState
};

pub mod optimizer;
pub use crate::optimizer::{ArcVecArc, EmptyInfo, OptInfo, Optimizer, VecArc};

pub mod stop;
pub use stop::Stop;

pub mod experiment;

pub mod recorder;
pub use recorder::{CSVRecorder};

pub mod checkpointer;