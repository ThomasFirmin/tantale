//! # Tantale core
//!
//! `tantale_core` provides typed building blocks for black-box optimization:
//! domains, search spaces, solutions, objective outputs, samplers, stopping
//! criteria, experiment runners, recording and checkpointing traits.
//!
//! ## Install
//!
//! Add the core crate when implementing integrations or custom components:
//!
//! ```toml
//! [dependencies]
//! tantale_core = "0.3.2"
//! ```
//!
//! For the macros and built-in algorithms used in the examples below, depend on
//! the facade crate instead:
//!
//! ```toml
//! [dependencies]
//! tantale = "0.3.2"
//! ```
//!
//! The workspace uses Rust 1.91.1 or later and the Rust 2024 edition. The
//! `mpi` feature enables MPI communication types. Python bindings are provided by the separate
//! `tantale_python` crate and are not part of `tantale_core`.
//!
//! ## Architecture
//!
//! A problem connects an input model, a function, and the interpretation of
//! that function's output:
//!
//! 1. A [`Domain`] defines valid values. A [`Var`] pairs the function-facing
//!    domain with the optimizer-facing domain. When those representations
//!    differ, [`Onto`] maps between them.
//! 2. A [`Searchspace`] collects variables and samples or maps complete
//!    candidate inputs. [`Sp`] is the standard implementation.
//! 3. An [`Outcome`] contains the raw result of evaluating a candidate. Its
//!    [`Codomain`] extracts objective values, constraints, or cost. A
//!    single-objective result and a multi-objective result use different
//!    codomain types.
//! 4. A [`Problem`] groups the search space and function. [`Pb`] is the
//!    general implementation; [`SimplePb`] and [`FidPb`] are aliases for
//!    one-step and multi-fidelity problems.
//! 5. A [`Sampler`] proposes candidates and receives computed solutions to
//!    update its state. Algorithms such as random search are provided by
//!    `tantale_algos`; the core crate defines the interface, not a separate
//!    samplers.
//! 6. An [`Experiment`] runs a problem with a sampler and a [`Stop`] criterion.
//!    [`ParExperiment`] uses a [`utils::Communicator`] to coordinate workers. The
//!    [`ThreadCommunicator`] implementation is available without features;
//!    MPI communication types require `mpi`.
//!
//! Experiment results are collected in a codomain-specific accumulator. A
//! single-objective problem keeps its best result; multi-objective problems
//! maintain a Pareto set.
//!
//! ## Main modules
//!
//! - [`domain`]: input domains, domain mappings, and objective codomains.
//! - [`distribution`]: random sampling distributions used by domains.
//! - [`variable`]: variable definitions relating objective and optimizer domains.
//! - [`searchspace`]: variable collections and candidate sampling/mapping.
//! - [`solution`]: solution shapes, identifiers, metadata, and computed results.
//! - [`objective`]: function wrappers, outcomes, and multi-fidelity steps.
//! - [`optimizer`]: the [`Sampler`] interface and optimizer metadata.
//! - [`stop`]: [`Calls`], [`Evaluated`], [`stop::Time`], and the [`Stop`] trait.
//! - [`experiment`]: [`Experiment`], [`ParExperiment`], and communication.
//! - [`recorder`]: result persistence, including [`CSVRecorder`].
//! - [`checkpointer`]: experiment persistence, including [`MessagePack`].
//! - [`errors`]: errors returned by recorder, checkpointer, and communication
//!   components.
//!
//! ## Python integration
//!
//! The Python bridge is provided by the separate `tantale_python` crate and
//! exposed through the facade crate's `tantale::python` module. A Rust host
//! defines the search space and uses `init_python!` to load a Python objective
//! and outcome class. The Python function receives a candidate solution
//! in search-space order and returns an outcome object. The index constant is
//! generated from the Rust search-space variable name; this example assumes a
//! variable named `a`:
//!
//! ```python
//! from pytantale import indices as idx
//!
//! class Score:
//!     def __init__(self, value: float):
//!         self.value = value
//!
//!     @staticmethod
//!     def header():
//!         return ["value"]
//!
//!     def record(self):
//!         return [str(self.value)]
//!
//! def objective(x: list) -> Score:
//!     return Score(float(x[idx.A]) ** 2)
//! ```
//!
//! The outcome class supplies `header()` and `record()` for CSV
//! recording. The Rust-side `init_python!` declaration selects which outcome
//! fields are objectives.
//!
//! ## Single-shot optimization
//!
//! The facade crate's [`objective!`](../../target/doc/macros/macro.objective.html) macro declares the variables and wraps the
//! function. The `Outcome` derive marks which field to maximize or minimize
//! and generates the corresponding codomain. The generated `get_problem`
//! helper can then be passed to an experiment.
//!
//! This example runs random search, records CSV output, checkpoints state,
//! and removes its temporary output directory when complete.
//!
//! ```no_run
//! use tantale::{
//!     algos::RandomSearch,
//!     core::{
//!         Evaluated, CSVRecorder, 
//!         Experiment, FolderConfig, MessagePack,
//!         experiment::Runable,
//!     },
//! };
//! 
//! mod model {
//!     use super::*;
//!     use serde::{Deserialize, Serialize};
//!     use tantale::core::{Real, Uniform};
//!     use tantale::macros::{objective, Outcome, Recordable, Schema};
//! 
//!     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
//!     pub struct Score {
//!         #[maximize]
//!         pub value: f64,
//!     }
//!
//!     objective!(
//!         pub fn quadratic() -> Score {
//!             let x = [! x | Real(-2.0, 2.0, Uniform) | !];
//!             Score {
//!                 value: -(x - 0.25).powi(2),
//!             }
//!         }
//!     );
//! }
//! let problem = model::get_problem();
//! let config = FolderConfig::new("tmp_test_quadratic");
//! let recorder = CSVRecorder::new(config.clone());
//! let checkpointer = MessagePack::new(config);
//! 
//! let experiment = Experiment::new(
//!     problem,
//!     RandomSearch::new(),
//!     Evaluated::new(8),
//!     recorder,
//!     checkpointer,
//! );
//! let results = experiment.run();
//! let best = results.best.expect("No best solution found");
//! println!("Best solution: {:?}", best);
//! ```
//!
//! ## Multi-fidelity objectives
//!
//! A multi-fidelity function returns `(Outcome, State)` and marks the outcome's
//! [`Step`] field with `#[step]`. On a later evaluation of that candidate,
//! `[! STATE !]` provides the state returned by the previous call. Implement
//! [`FuncState`] to save and load that state when checkpointing is needed.
//! Unlike `Outcome`, `FuncState` does not prescribe a serialization format;
//! its `save` and `load` methods return `std::io::Result`.
//!
//! ```no_run
//!
//! mod training {
//!     use serde::{Deserialize, Serialize};
//!     use std::{io, path::PathBuf};
//!     use tantale::core::{EvalStep, FidOutcome, Outcome, SingleCodomain, Step, Real, Uniform, FuncState, EmptyInfo};
//!     use tantale::macros::{objective, Outcome, Recordable, Schema};
//!     
//!     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
//!     pub struct TrainingOutcome {
//!         #[maximize]
//!         score: f64,
//!         #[step]
//!         step: Step,
//!     }
//! 
//!     #[derive(Serialize, Deserialize)]
//!     pub struct TrainingState {
//!         epochs: u32,
//!     }
//! 
//!     impl FuncState for TrainingState {
//!         fn save(&self, path: PathBuf) -> io::Result<()> {
//!             std::fs::write(path.join("epochs"), self.epochs.to_le_bytes())
//!         }
//!         fn load(path: PathBuf) -> io::Result<Self> {
//!             let bytes: [u8; 4] = std::fs::read(path.join("epochs"))?
//!                 .try_into()
//!                 .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid epoch state"))?;
//!             Ok(Self {
//!                 epochs: u32::from_le_bytes(bytes),
//!             })
//!         }
//!     }
//! 
//!     objective!(
//!         pub fn train() -> (TrainingOutcome, TrainingState) {
//!             let learning_rate = [! learning_rate | Real(0.001, 0.1, Uniform) | !];
//!             let mut state = [! STATE !].unwrap_or(TrainingState { epochs: 0 });
//!             state.epochs += 1;
//!
//!             let step = if state.epochs == 3 {
//!                 Step::Evaluated
//!             } else {
//!                 Step::Partially(state.epochs as isize)
//!             };
//!             (
//!                 TrainingOutcome {
//!                     score: -learning_rate,
//!                     step,
//!                 },
//!                 state,
//!             )
//!         }
//!     );
//! }
//! 
//! // The EmptyInfo type is unecessary when problem is combined with a sampler and run, allowing the compiler to infer the type.
//! // It is included here for clarity.
//! let _problem = training::get_problem::<tantale::core::EmptyInfo>();
//! ```
//!
//! ## Parallel execution
//!
//! `ParExperiment` takes the same problem, sampler, stop criterion, recorder,
//! and checkpointer as `Experiment`, plus a communicator.
//! This example uses the thread communicator. For process-based distributed
//! execution, enable `mpi` and use the MPI communicator APIs after initializing
//! the MPI environment.
//!
//! ```no_run
//! use tantale::{
//!     algos::RandomSearch,
//!     core::{
//!         Evaluated, CSVRecorder, FolderConfig, MessagePack, ParExperiment,
//!         experiment::{Runable, ThreadCommunicator},
//!         Outcome,
//!     },
//! };
//! use serde::{Deserialize, Serialize};
//! use tantale::macros::{Outcome, objective, Recordable, Schema};
//!
//! mod model {
//!     use super::*;
//!     use serde::{Deserialize, Serialize};
//!     use tantale::core::{Real, Uniform};
//!     use tantale::macros::{Outcome, objective, Recordable, Schema};
//!
//!     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
//!     pub struct Score {
//!         #[maximize]
//!         pub value: f64,
//!     }
//!
//!     objective!(
//!         pub fn quadratic() -> Score {
//!             let x = [! x | Real(-2.0, 2.0, Uniform) | !];
//!             Score {
//!                 value: -(x - 0.25).powi(2),
//!             }
//!         }
//!     );
//! }
//!
//! let problem = model::get_problem();
//! let config = FolderConfig::new("threaded-run");
//! let experiment = ParExperiment::new(
//!     problem,
//!     RandomSearch::new(),
//!     Evaluated::new(32),
//!     CSVRecorder::new(config.clone()),
//!     MessagePack::new(config),
//!     ThreadCommunicator::new(),
//! );
//! let _results = experiment.run();
//! ```
//!
//! ## Configuration, stopping, and persistence
//!
//! [`FolderConfig::new`] gives paths for the run, recorder, and checkpointer;
//! the experiment components create their output directories when initialized.
//! [`CSVRecorder::new`] enables CSV recording, and [`MessagePack::new`] enables
//! MessagePack checkpointing. Both constructors currently return `Option`.
//!
//! [`Calls::new`] stops after a certain number of calls to the objective function. 
//! [`Evaluated::new`] counts only fully evaluated candidates. [`stop::Time::new`] sets a duration limit. 
//! Each criterion implements [`Stop`] and can be extended with its `add` method.
//!
//! The objective function returns an `Outcome`, not a `Result`. Recorder and
//! checkpointer traits expose typed errors, but the experiment runner currently
//! unwraps some lifecycle errors, so `Runable::run` does not return those
//! errors to the caller.
//!
//! ## Extending the core
//!
//! Implement [`Domain`] and [`Searchspace`] types to add input representations,
//! [`Outcome`] to define an output and codomain, [`Sampler`] to add an
//! optimization strategy, and [`Stop`], [`Recorder`], or [`Checkpointer`] to
//! customize execution. The core types and traits are re-exported at the
//! crate root; their defining modules provide more specific documentation.

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

impl State for GlobalParameters {}

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
    Criteria, Domain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain,
    ElemCostConstCodomain, ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain,
    ElemSingleCodomain, Grid, GridDom, GridInt, GridNat, GridReal, Int, Linked, Mixed,
    MixedTypeDom, Multi, MultiCodomain, Nat, NoDomain, Onto, OntoDom, ParetoAccumulator, Real,
    Single, SingleCodomain, Unit,
};

pub mod distribution;
pub use distribution::{
    Bernoulli, BoolDistribution, BoundedDistribution, Distribution, GridDomDistribution, Uniform,
};

pub mod variable;
pub use variable::var::Var;

pub mod solution;
pub use solution::{
    BaseSol, Computed, Fidelity, FidelitySol, Lone, Pair, ParSId, SId, Sampled, SolId, SolInfo,
    Solution, SolutionShape, StepId, StepSId, Uncomputed, IntoComputedShape, UncomputedShape,
};

pub mod searchspace;
pub use searchspace::{Searchspace, Sp, SpPar};

pub mod errors;

pub mod objective;
pub use crate::objective::{
    EvalStep, FidOutcome, FuncWrapper, MultiStep, OneStep, Outcome, Step, outcome::FuncState,
};

pub mod optimizer;
pub use crate::optimizer::{EmptyInfo, Prunable, Pruner, Sampler};

pub mod stop;
pub use stop::{Calls, Evaluated, Stop};

pub mod experiment;
pub use experiment::{
    Experiment, ParExperiment, PoolMode, Runable, ThreadChannel, ThreadCommunicator,
};
#[cfg(feature = "mpi")]
pub use experiment::{MPIChannel, MPICommunicator};

pub mod recorder;
pub use recorder::{CSVRecorder, Recorder};

pub mod checkpointer;
pub use checkpointer::{Checkpointer, MessagePack};

pub mod utils;
pub use utils::{
    Dominate, DominateView, FolderConfig, HasFidelity, HasId, HasInfo, HasState, HasStep,
    HasStepId, HasUncomputed, HasVariables, HasX, HasY, IntoNonDominatedSorting, IntoParetoFront,
    NdArrayDominate, NonDominatedSorting, Orderable, OrderedArchive, ParetoFront, Recordable,
    Schema, State, XToNdArray, Xy, YToNdArray, aliases,
};

#[cfg(feature = "spikes")]
pub mod spikes;
#[cfg(feature = "spikes")]
pub use spikes::*;

pub mod problem;
pub use problem::{FidPb, Pb, Problem, SimplePb};
