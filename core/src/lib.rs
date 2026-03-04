//! # Core
//!
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
//! - **Optimization:** [`Optimizer`], [`BatchOptimizer`], [`SequentialOptimizer`]
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
//! - [`serde`]: Most core types implement [`Serialize`](serde::Serialize) and
//!   [`Deserialize`](serde::Deserialize) to support checkpointing.
//! - [`rand`]: Domains and samplers use RNG traits to generate valid values.
//! - `mpi` (optional): Distributed experiments are enabled behind the `mpi`
//!   feature flag and expose MPI-specific execution helpers.
//! - [`rayon`]: Parallel evaluation and orchestration in multi-threaded runs. Used in synchrnous batched optimization, and [`SpPar`](crate::searchspace::SpPar) utilities.
//! - `csv`: CSV-backed [`Recorder`], with [`CSVRecorder`](crate::recorder::CSVRecorder) as the main implementation.
//! - `rmp-serde`: MessagePack serialization used by the default checkpointer. Used by [`MessagePack`](crate::checkpointer::MessagePack) for compact checkpointing of experiment state.
//! - `bincode`: Compact binary serialization for checkpointing and transport.
//!   Used by MPI-distributed optimization to create messages from [`Uncomputed`](crate::solution::Uncomputed) and [`Outcome`](crate::objective::Outcome).
//! - `bitvec`: Efficient bit-level storage in domain and solution utilities.
//!   Used by MPI-distributed optimization within [`IdleWorker`](crate::experiment::mpi::utils::IdleWorker) to track idle workers.
//! - `num_cpus`: Detects available CPU cores to size thread pools. Used in multi-threaded execution contexts, for asynchrnous multi-threaded optimization, to determine pool size of threads.
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
//! can be evaluated by [`Step`](crate::objective::Step). Hence, the function must maintain an internal state that is updated at each step.
//! To trigger the wrapping within a [`Stepped`] objective, the user must defined an [`Outcome`] containing a [`Step`](crate::objective::Step).
//!
//! ```rust
//! use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform, Step};
//! use tantale::macros::{objective, Outcome, CSVWritable};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Outcome, CSVWritable, Debug, Serialize, Deserialize)]
//! struct OutExample {
//!     obj: f64,
//!     info: f64,
//!     step: Step,
//! }
//!
//! #[derive(FuncState, Serialize, Deserialize)]
//! pub struct FnState {
//!     pub something: isize,
//! }
//!
//! objective!(
//!     pub fn example() -> (OutExample, FnState) {
//!         let _a = [! a | Int(0,100, Uniform) | !];
//!         let _b = [! b | Nat(0,100, Uniform) | !];
//!         let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
//!         let _d = [! d | Bool(Bernoulli(0.5)) | !];
//!         let e = [! e | Real(1000.0,2000.0, Uniform) | !];
//!         // ... more variables and computation ...
//!         
//!         // Manage the internal state
//!         state.something += 1;
//!         let evalstate = if state.something == 5 {Step::Evaluated} else{Step::Partially(state.something)};
//!         
//!         (
//!             OutExample{
//!                 obj: e,
//!                 info: [! f | Real(10.0,20.0, Uniform) | !],
//!                 step: evalstate,
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
//! use tantale::algos::{random_search, BatchRandomSearch};
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = BatchRandomSearch::new(7);
//! let cod: SingleCodomain<_> = random_search::codomain(|o: OutExample| o.obj);
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_batch").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = mono((sp, cod), obj, opt, stop, (rec, check));
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
//! use tantale::algos::{random_search, RandomSearch};
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = RandomSearch::new();
//! let cod = random_search::codomain(|o: OutExample| o.obj);
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_seq_threads").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
//! exp.run();
//! ```
//!
//! ### Multi-fidelity batch run
//!
//! ```rust,ignore
//! use tantale::core::{
//!     CSVRecorder, FolderConfig, MessagePack, experiment::{Runable, mono}, stop::Calls,
//! };
//! use tantale::algos::{random_search,BatchRandomSearch};
//!
//! let sp = my_module::get_searchspace();
//! let obj = my_module::get_function();
//! let opt = BatchRandomSearch::new(7);
//! let cod = random_search::codomain(|o: OutExample| o.obj);
//!
//! let stop = Calls::new(50);
//! let config = FolderConfig::new("run_fidelity").init();
//! let rec = CSVRecorder::new(config.clone(), true, true, true, true);
//! let check = MessagePack::new(config);
//!
//! let exp = mono((sp, cod), obj, opt, stop, (rec, check));
//! exp.run();
//! ```

use serde::{Deserialize, Serialize};
#[cfg(feature = "mpi")]
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

/// Global counter for solution identifiers.
pub static SOL_ID: AtomicUsize = AtomicUsize::new(0);
/// Global counter for optimizer identifiers.
pub static OPT_ID: AtomicUsize = AtomicUsize::new(0);
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
    /// Last optimizer id.
    pub opt_id: usize,
    /// Last run id.
    pub run_id: usize,
}

pub mod config;
#[cfg(feature = "mpi")]
pub use config::DistSaverConfig;
pub use config::{FolderConfig, SaverConfig};

pub mod domain;
pub use domain::{
    Bool, Bounded, Cat, Codomain, ConstCodomain, ConstMultiCodomain, Constrained, Cost,
    CostCodomain, CostConstCodomain, CostConstMultiCodomain, CostMultiCodomain, Criteria, Domain,
    FidCriteria, HasEvalStep, Int, LinkObj, LinkOpt, LinkTyObj, LinkTyOpt, Mixed, MixedTypeDom,
    Multi, MultiCodomain, Nat, NoDomain, Onto, Real, Single, SingleCodomain, Unit,
};

pub mod sampler;
pub use sampler::{
    Bernoulli, BoolDistribution, BoundedDistribution, CatDistribution, Sampler, Uniform,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{
    BaseSol, Batch, Computed, Fidelity, FidelitySol, HasFidelity, HasId, HasInfo, HasSolInfo,
    HasStep, HasUncomputed, HasY, Id, IntoComputed, OutBatch, ParSId, SId, SolInfo, Solution,
    SolutionShape, shape::RawObj,
};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp, SpPar};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    EvalStep, FidOutcome, FuncWrapper, Objective, Outcome, Step, Stepped, outcome::FuncState,
};

pub mod optimizer;
pub use crate::optimizer::{
    BatchOptimizer, EmptyInfo, OptInfo, OptState, Optimizer, SequentialOptimizer, opt::CompBatch,
};

pub mod stop;
pub use stop::{Calls, Stop};

pub mod experiment;
#[cfg(feature = "mpi")]
pub use experiment::{
    DistEvaluate, Evaluate, MPIExperiment, MasterWorker, distributed, mpi::utils::MPIProcess,
    mpi::worker::Worker,
};
pub use experiment::{MonoExperiment, Runable, ThrExperiment, mono, threaded};

pub mod recorder;
#[cfg(feature = "mpi")]
pub use recorder::DistRecorder;
pub use recorder::{CSVRecorder, CSVWritable, Recorder};

pub mod checkpointer;
#[cfg(feature = "mpi")]
pub use checkpointer::DistCheckpointer;
pub use checkpointer::{Checkpointer, MessagePack, MonoCheckpointer, ThrCheckpointer};
