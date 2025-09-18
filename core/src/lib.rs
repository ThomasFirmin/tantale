//! # Core
//!
//! This the core of the library containing most of the submodules, and basic software bricks.

use std::sync::atomic::AtomicUsize;
#[cfg(feature="mpi")]
use std::sync::OnceLock;
use serde::{Serialize,Deserialize};


pub static SOL_ID: AtomicUsize = AtomicUsize::new(0);
pub static OPT_ID: AtomicUsize = AtomicUsize::new(0);
pub static RUN_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Serialize, Deserialize)]
pub struct GlobalParameters {
    pub sold_id: usize,
    pub opt_id: usize,
    pub run_id: usize,
}

#[cfg(feature="mpi")]
pub static MPI_UNIVERSE: OnceLock<mpi::environment::Universe> = OnceLock::new();
#[cfg(feature="mpi")]
pub static MPI_WORLD: OnceLock<mpi::topology::SimpleCommunicator> = OnceLock::new();
#[cfg(feature="mpi")]
pub static MPI_SIZE: OnceLock<mpi::Rank> = OnceLock::new();
#[cfg(feature="mpi")]
pub static MPI_RANK: OnceLock<mpi::Rank> = OnceLock::new();

#[cfg(feature="mpi")]
pub fn mpi_init(){
    use mpi::traits::Communicator;

    if MPI_UNIVERSE.get().is_none(){
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size();
        let rank = world.rank();
        MPI_UNIVERSE.set(universe);
        MPI_WORLD.set(world);
        MPI_SIZE.set(size).unwrap();
        MPI_RANK.set(rank).unwrap();
    }
    else{
        panic!("The MPI Universe has already been initialized")
    }
}

pub mod domain;
pub use domain::{
    uniform, uniform_bool, uniform_cat, uniform_int, uniform_nat, uniform_real, BaseDom,
    BaseTypeDom, Bool, Bounded, Cat, Domain, DomainBoundariesError, DomainBounded, DomainError,
    DomainOoBError, Int, Nat, Onto, Real, Unit,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{Computed, Id, ParSId, Partial, SId, SolInfo, Solution,FidelState,FidelityInfo};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Criteria, FidelCodomain,
    FidelConstCodomain, FidelConstMultiCodomain, FidelMultiCodomain, Fidelity, LinkedOutcome,
    Multi, MultiCodomain, Objective, Stepped, Outcome, Single, SingleCodomain,
};

pub mod optimizer;
pub use crate::optimizer::{ArcVecArc, EmptyInfo, OptInfo, Optimizer};

pub mod stop;
pub use stop::Stop;

pub mod experiment;

pub mod saver;
pub use saver::CSVSaver;
