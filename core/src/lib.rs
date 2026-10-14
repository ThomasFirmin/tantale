//! Tantale core contains the foundational abstractions and building blocks used
//! throughout the library. It defines the modeling layer (domains, variables,
//! search spaces, solutions), the optimization layer (objectives, optimizers,
//! stop criteria), and the execution layer (experiments, recorders,
//! checkpointing).
//!
//! ## Main building blocks
//!
//! - **Modeling:** [`Domain`], [`Var`], [`Searchspace`], [`Solution`]
//! - **Objective:** [`Objective`], [`Outcome`], [`Stepped`]
//! - **Optimization:** [`Optimizer`], [`BatchOptimizer`], [`SingleOptimizer`]
//! - **Execution:** [`Runable`], [`Recorder`], [`Checkpointer`], [`Stop`]
//!
//! Most of these items are re-exported at the crate root for convenience.
//!
//! ## Module map
//!
//! - [`domain`]: Domain definitions and codomains (objective outputs).
//! - [`sampler`]: Sampling utilities and distributions.
//! - [`variable`]: [`Var`] to relate objective and optimizer domains.
//! - [`searchspace`]: Composition of variables into a [`Searchspace`].
//! - [`solution`]: Solution shapes, identifiers, solution metadata, and computed wrappers.
//! - [`objective`]: Objective functions, steps, and outcomes.
//! - [`optimizer`]: Optimizer traits and state metadata.
//! - [`stop`]: Stop criteria and lifecycle signals.
//! - [`experiment`]: Execution pipelines and orchestration.
//! - [`recorder`]: Recording utilities (e.g., CSV).
//! - [`checkpointer`]: Checkpointing primitives.
//!
//! ## Execution contexts
//!
//! Tantale provides multiple execution contexts through [`Runable`]:
//! - [`MonoExperiment`] for single-threaded runs.
//! - [`ThrExperiment`] for multi-threaded runs.
//! - [`MPIExperiment`] for distributed runs (requires the `mpi` feature).
//!
//! ## Feature flags
//!
//! - `mpi`: Enables MPI-based distributed execution and related utilities.
//!
//! ## Dependencies and integration
//!
//! Tantale core relies on a few external crates:
//! - `serde`: Most core types implement [`Serialize`] and
//!   [`Deserialize`] to support checkpointing.
//! - `rand`: Domains and samplers use RNG traits to generate valid values.
//! - `mpi` (optional): Distributed experiments are enabled behind the `mpi`
//!   feature flag and expose MPI-specific execution helpers.
//! - `rayon`: Parallel evaluation and orchestration in multi-threaded runs. Used in synchrnous batched optimization, and [`SpPar`] utilities.
//! - `csv`: CSV-backed [`Recorder`], with [`CSVRecorder`] as the main implementation.
//! - `rmp-serde`: MessagePack serialization used by the default checkpointer. Used by [`MessagePack`] for compact checkpointing of experiment state.
//! - `bincode`: Compact binary serialization for checkpointing and transport.
//!   Used by MPI-distributed optimization to create messages from [`Uncomputed`] and [`Outcome`].
//! - `bitvec`: Efficient bit-level storage in domain and solution utilities.
//!   Used by MPI-distributed optimization within [`IdleWorker`](crate::experiment::mpi::utils::IdleWorker) to track idle workers.
//! - `num_cpus`: Detects available CPU cores to size thread pools. Used in multi-threaded execution contexts, for asynchrnous multi-threaded optimization, to determine pool size of threads.
//! - `indexmap`: Provides a hashmap preserving iteration order. It used to managa
//!   function states, for FIFO discarding of old states to save memory.
//!
//! ## Examples
//!
//! ### Define a searchspace with `objective!`
//!
//! The `objective!` macro builds a [`Searchspace`] and wraps a user-defined
//! function into an [`Objective`]. The example below mirrors the evaluator
//! pattern used in the test suite.
//!
//! The objective macro and the function must be defined within another module than the main function, to avoid issues with the generated code.
//! The macro generates a searchspace based on the variables defined in the function body, and an objective that wraps the user function.
//! The searchspace and objective can then be created via generated function: `my_module::get_searchspace()` and `my_module::get_function()`.
//!
//! ```rust
//! use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
//! use tantale::macros::{objective, Outcome, CSVWritable};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
//! struct OutExample {
//!     #[maximize]
//!     obj: f64,
//!     info: f64,
//! }
//!
//! objective!(
//!     pub fn example() -> OutExample {
//!         let _a = [! a | Int(0,100, Uniform) | !];
//!         let _b = [! b | Nat(0,100, Uniform) | !];
//!         let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
//!         let _d = [! d | Bool(Bernoulli(0.5)) | !];
//!         let e = [! e | Real(1000.0,2000.0, Uniform) | !];
//!         // ... more variables and computation ...
//!         OutExample{
//!             obj: e,
//!             info: [! f | Real(10.0,20.0, Uniform) | !],
//!         }
//!     }
//! );
//!
//! // The macro expands to helpers like:
//! // let sp = example::get_searchspace();
//! // let obj = example::get_function();
//! ```
//!
//! ### Define a searchspace with `objective!` for [`Stepped`] function
//!
//! [`Stepped`] functions is similar to [`Objective`] except that the user-defined function
//! can be evaluated by [`Step`]. Hence, the function must maintain an internal state that is updated at each step.
//! To trigger the wrapping within a [`Stepped`] objective, the user must defined an [`Outcome`] containing a [`Step`].
//!
//! First, the user defines the function state by implementing the [`FuncState`] trait,
//! which requires implementing the `save` and `load` methods for checkpointing.
//! Then, the user defines an outcome struct that contains a field of type `Step`.
//! Finally, the user can use the `objective!` macro to define the function, and manage the internal state and evaluation step within the function body.
//!
//! ```rust
//! use tantale::core::{FuncState, Bool, Cat, Int, Nat, Real, Bernoulli, Uniform, Step};
//! use tantale::macros::{objective, Outcome, CSVWritable};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
//! struct OutExample {
//!     #[maximize]
//!     obj: f64,
//!     info: f64,
//!     #[step]
//!     step: Step,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! pub struct FnState {
//!     pub something: isize,
//! }
//!
//! // The user must implement the saving and loading logic of the function state,
//! // to enable checkpointing of the internal state of the function.
//! // The input is a folder path where the checkpoint is stored, and the user can decide how to serialize the state (e.g., using rmp_serde or bincode).
//! impl FuncState for FnState {
//!    fn save(&self, path: std::path::PathBuf) -> std::io::Result<()>{
//!        let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
//!        rmp_serde::encode::write(&mut file, &self).unwrap();
//!        Ok(())
//!    }
//!    fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
//!        let file_path = path.join("fn_state.mp");
//!        let file = std::fs::File::open(file_path)?;
//!        let state = rmp_serde::decode::from_read(file).unwrap();
//!        Ok(state)
//!    }
//! }
//!
//! objective!(
//!     pub fn example() -> (OutExample, FnState) {
//!         let _a = [! a | Int(0,100, Uniform) | !];
//!         let _b = [! b | Nat(0,100, Uniform) | !];
//!         let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
//!         let _d = [! d | Bool(Bernoulli(0.5)) | !];
//!         let e = [! e | Real(1e-4, 1e-1, Uniform) | Log !];
//!         // ... more variables and computation ...
//!         
//!         // Manage the internal state
//!         let mut state = if let Some(s) = [! STATE !] {
//!             s
//!         } else {
//!             FnState { something: 0 }
//!         };
//!         state.something += 1;
//!         let evalstep = if state.something == 5 {Step::Evaluated} else{Step::Partially(state.something)};
//!         
//!         (
//!             OutExample{
//!                 obj: e,
//!                 info: [! f | Real(10.0,20.0, Uniform) | !],
//!                 step: evalstep,
//!             },
//!             state
//!         )
//!     }
//! );
//! ```
//!
//! ### Batch run with checkpointing (mono)
//!
//! ```rust,ignore
//! use tantale::core::{
//!     CSVRecorder, FolderConfig, MessagePack, Objective, SingleCodomain,
//!     experiment::{Runable, mono}, stop::Calls,
//! };
//! use tantale::algos::BatchRandomSearch;
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = BatchRandomSearch::new(7);
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_batch").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = mono(sp, obj, opt, stop, (rec, check));
//! exp.run();
//! ```
//!
//! ### Sequential run with threaded execution
//!
//! ```rust,ignore
//! use tantale::core::{
//!     CSVRecorder, FolderConfig, MessagePack, Objective,
//!     experiment::{Runable, threaded}, stop::Calls,
//! };
//! use tantale::algos::RandomSearch;
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = RandomSearch::new();
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_seq_threads").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = threaded(sp, obj, opt, stop, (rec, check));
//! exp.run();
//! ```
//!
//! ### Multi-fidelity batch run
//!
//! ```rust,ignore
//! use tantale::core::{
//!     CSVRecorder, FolderConfig, MessagePack, experiment::{Runable, mono}, stop::Calls,
//! };
//! use tantale::algos::{random_search::RandomSearch};
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = BatchRandomSearch::new(7);
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_fidelity").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = mono(sp, obj, opt, stop, (rec, check));
//! exp.run();
//! ```

use serde::{Deserialize, Serialize};
#[cfg(feature = "mpi")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter for solution identifiers.
pub static SOL_ID: AtomicUsize = AtomicUsize::new(0);
/// Global counter for run identifiers.
pub static RUN_ID: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "mpi")]
pub static MPI_RANK: OnceLock<mpi::Rank> = OnceLock::new();
#[cfg(feature = "mpi")]
pub static MPI_SIZE: OnceLock<mpi::Rank> = OnceLock::new();

/// Serializable snapshot of global counters for checkpointing.
#[derive(Serialize, Deserialize)]
pub struct GlobalParameters {
    /// Last solution id.
    pub sold_id: usize,
    /// Last run id.
    pub run_id: usize,
}

impl State for GlobalParameters { }

impl HasState for GlobalParameters {
    type State = Self;

    fn get_mut_state(&mut self) -> &mut Self::State {
        self
    }

    fn get_state(&self) -> &Self::State {
        self
    }

    fn from_state(state: Self::State) -> Self {
        state
    }
}

impl GlobalParameters {
    /// Create a new instance of `GlobalParameters` with default values.
    pub fn new() -> Self {
        Self {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        }
    }
}

impl Default for GlobalParameters {
    fn default() -> Self {
        Self::new()
    }
}

extern crate self as tantale_core;

pub mod domain;
pub use domain::{
    Accumulator, BestAccumulator, Bool, Bounded, Cat, Codomain, ConstCodomain, ConstMultiCodomain,
    Constrained, Cost, CostCodomain, CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain,
    Domain, Grid, GridDom, GridInt, GridNat, GridReal, Int,
    Linked, Mixed, MixedTypeDom, Multi, MultiCodomain, Nat,
    NoDomain, Onto, OntoDom, ParetoAccumulator, Real, Single, SingleCodomain,
    Unit,
    ElemSingleCodomain,
    ElemCostCodomain,
    ElemConstCodomain,
    ElemCostConstCodomain,
    ElemMultiCodomain,
    ElemCostMultiCodomain,
    ElemConstMultiCodomain,
    ElemCostConstMultiCodomain,
    Criteria,
};

pub mod distribution;
pub use distribution::{
    Bernoulli, BoolDistribution, BoundedDistribution, Distribution, GridDomDistribution, Uniform,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{
    BaseSol, Computed, Fidelity, FidelitySol, SolId, Lone,
    Pair, ParSId, SId, SolInfo, Sampled, Solution, SolutionShape, StepId, StepSId, Uncomputed,
};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp, SpPar};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    EvalStep, FidOutcome, FuncWrapper, OneStep, Outcome, Step, MultiStep, outcome::FuncState,
};

pub mod optimizer;
pub use crate::optimizer::{EmptyInfo, Sampler, Pruner, Prunable};

pub mod stop;
pub use stop::{Calls, Evaluated, Stop};

pub mod experiment;
pub use experiment::{
    Experiment, PoolMode, Runable,
};

pub mod recorder;
pub use recorder::{CSVRecorder, Recorder};

pub mod checkpointer;
pub use checkpointer::{Checkpointer, MessagePack};

pub mod utils;
pub use utils::{
    Dominate, HasFidelity, HasId, HasInfo, HasStep, HasStepId, HasUncomputed,
    HasVariables, HasX, HasY, IntoNonDominatedSorting, IntoParetoFront, NdArrayDominate,
    NonDominatedSorting, Orderable, OrderedArchive, ParetoFront, XToNdArray, Xy, YToNdArray, State,
    HasState, FolderConfig, Recordable, Schema,
    aliases,
};


#[cfg(feature = "spikes")]
pub mod spikes;
#[cfg(feature = "spikes")]
pub use spikes::*;

pub mod problem;
pub use problem::{Problem, SimplePb, FidPb, Pb};

