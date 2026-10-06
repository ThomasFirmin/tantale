//! Experiment orchestration and optimization loops.
//!
//! This module provides the complete infrastructure for running optimization experiments in Tantale.
//! It combines all essential components—[`Searchspace`], [`Optimizer`], objective functions,
//! stopping criteria, recorders, and checkpointers—into cohesive execution pipelines.
//!
//! # Core Concept: The Optimization Pipeline
//!
//! An optimization experiment in Tantale follows this general workflow:
//!
//! ```text
//! -------------------
//! |   Initialize    |  <- Load from checkpoint or start fresh
//! -------------------
//!          |
//!          Y
//! -------------------
//! |  Generate       |  <- Optimizer produces candidate solutions
//! |  Solutions      |
//! -------------------
//!          |
//!          Y
//! -------------------
//! |  Evaluate       |  <- Objective function evaluates solutions
//! |  Objective      |
//! -------------------
//!          |
//!          Y
//! -------------------
//! |  Update         |  <- Optimizer updates its internal state
//! |  Optimizer      |
//! -------------------
//!          |
//!          Y
//! -------------------
//! |  Record &       |  <- Save results and checkpoint state
//! |  Checkpoint     |
//! -------------------
//!          |
//!          Y
//! -------------------
//! |  Check Stop     |  <- Evaluate stopping criterion
//! |  Criterion      |
//! -------------------
//!          |
//!     No   |   Yes
//!   ---------------
//!   |             |
//!   Y             Y
//! Loop       Terminate
//! ```
//!
//! ## Note
//!
//! This workflow is abstract and can be adapted to different execution contexts (single-threaded, multi-threaded, distributed)
//! and optimizer paradigms (batch vs sequential). The provided experiment types encapsulate these variations while adhering to the same core principles.
//!
//! # Experiment Types
//!
//! Tantale provides three execution contexts, each tailored to different parallelization strategies:
//!
//! ** The following bits of code are mock examples for documentation purposes.
//! They are not meant to be compiled or run as-is, but rather to illustrate the intended usage and API of the experiment module. **
//!
//! ## 1. [`MonoExperiment`] - Single-Threaded Execution
//!
//! Sequential execution in a single thread.
//!
//! ```rust,ignore
//! use tantale::core::experiment::mono;
//!
//! let experiment = mono(
//!     searchspace,
//!     objective,
//!     optimizer,
//!     stop_criterion,
//!     (Some(recorder), Some(checkpointer))
//! );
//! experiment.run();
//! ```
//!
//! ## 2. [`ThrExperiment`] - Multi-Threaded Execution
//!
//! Parallel execution using OS threads.
//!
//! ```rust,ignore
//! use tantale::core::experiment::threaded;
//!
//! let experiment = threaded(
//!     searchspace,
//!     objective,
//!     optimizer,
//!     stop_criterion,
//!     (Some(recorder), Some(checkpointer))
//! );
//! experiment.run();
//! ```
//!
//! ## 3. [`MPIExperiment`] - Distributed Execution (MPI feature)
//!
//! Distributed parallel execution across multiple processes using MPI.
//! Requires the `mpi` feature and an MPI environment.
//!
//! ```rust,ignore
//! use tantale::core::{experiment::{distributed,mpi::MPIProcess}, FolderConfig, MessagePack};
//!
//! // Initialize MPI process containing rank, communicator...
//! let mpi_process = MPIProcess::new();
//!
//! let config = FolderConfig::new("distributed_experiment").init(&mpi_process);
//! let checkpointer = MessagePack::new(config.clone());
//!
//! let experiment = distributed(
//!     &mpi_process,
//!     searchspace,
//!     objective,
//!     optimizer,
//!     stop_criterion,
//!     (Some(recorder), Some(checkpointer))
//! );
//! experiment.run();
//! ```
//!
//! # Optimizer Types
//!
//! Experiments adapt to two optimizer paradigms:
//!
//! ## Batch Optimization ([`BatchOptimizer`](crate::optimizer::BatchOptimizer))
//!
//! The optimizer generates a complete batch of solutions at each iteration:
//! - All solutions in a batch are evaluated simultaneously (parallelism across batch)
//! - Evaluation times are expected to be similar
//! - Examples: Random Search with batch size, CMA-ES population
//!
//! ## Sequential Optimization ([`SingleOptimizer`](crate::optimizer::SingleOptimizer))
//!
//! The optimizer generates solutions one at a time:
//! - Solutions are generated on-demand as evaluations complete
//! - Maximizes resource utilization when parallelized and evaluation times vary
//! - Examples: Bayesian Optimization (mono-thread sequential, not q-EI acquisition generating batches for example), Bayesian Optimization based on Thompson Sampling (asynchronous parallelism)
//!
//! # Objective Function Types
//!
//! ## Standard Objectives ([`Objective`](crate::objective::Objective))
//!
//! Single-shot evaluation: input → output
//! See the `objective!` procedural macro for easy definition.
//! ```rust,ignore
//!
//! // This is automatically generated and wrapped with the objective! procedural macro.
//! fn my_objective(x: Solution) -> MyOutcome {
//!     // Compute and return result
//! }
//! let obj = Objective::new(my_objective);
//! ```
//!
//! ## Multi-Fidelity Objectives ([`Stepped`](crate::objective::Stepped))
//!
//! Incremental evaluation with internal state ([`FuncState`](crate::FuncState)):
//! - Can be evaluated partially and resumed
//! - Useful for iterative algorithms (e.g., neural network training)
//! - Enables multi-fidelity optimization strategies
//!
//! See the `objective!` procedural macro for easy definition.
//!
//! ```rust,ignore
//! // This is automatically generated and wrapped with the objective! procedural macro.
//! fn training_step(x: Solution, state: TrainingState) -> (TrainingOutcome, TrainingState) {
//!     // Perform one training epoch
//! }
//! let stepped_obj = Stepped::new(training_step);
//! ```
//!
//! # Checkpointing and Recovery
//!
//! All experiment types support checkpointing for fault tolerance:
//!
//! ```rust,ignore
//! use tantale::core::{load, MessagePack, FolderConfig};
//!
//! // Create experiment
//! let config = FolderConfig::new("my_experiment").init();
//! let checkpointer = MessagePack::new(config);
//! let exp = mono(space, obj, opt, stop, (None, Some(checkpointer)));
//! exp.run();
//!
//! // Later: resume from checkpoint
//! let config = FolderConfig::new("my_experiment").init();
//! let checkpointer = MessagePack::new(config).unwrap();
//! let exp = load!(mono, MyOptimizer, MyStop, space, obj, (None, checkpointer));
//! exp.run();
//! ```
//!
//! # Recording Results
//!
//! Use [`Recorder`] to save evaluated solutions throughout optimization:
//!
//! ```rust,ignore
//! use tantale::core::CSVRecorder;
//!
//! let recorder = CSVRecorder::new(
//!     config,
//!     true,  // save objective domain
//!     true,  // save optimizer domain
//!     true,  // save codomain
//!     true   // save outcomes
//! );
//! ```
//!
//! # Stopping Criteria
//!
//! Control when optimization terminates using [`Stop`] implementations:
//!
//! - [`Calls`](crate::stop::Calls) - Stop after a number of function evaluations
//! - [`Time`](crate::stop::Time) - Stop after a duration
//! - Custom criteria implementing [`Stop`]
//!
//! Multiple criteria can be combined through custom implementations.
//!
//! # Complete Example
//!
//! ```rust,ignore
//! use tantale::core::*;
//! use tantale::algos::RandomSearch;
//!
//! // Define the problem
//! let searchspace = /* ... */;
//! let codomain = SingleCodomain::new(|out: &MyOutcome| out.value);
//! let objective = Objective::new(my_function);
//!
//! // Configure optimization
//! let optimizer = RandomSearch::new();
//! let stop = Calls::new(1000);
//!
//! // Setup saving
//! let config = FolderConfig::new("results").init();
//! let recorder = CSVRecorder::new(config.clone(), true, true, true, true);
//! let checkpointer = MessagePack::new(config);
//!
//! // Run experiment
//! let experiment = mono(
//!     searchspace,
//!     objective,
//!     optimizer,
//!     stop,
//!     (Some(recorder), Some(checkpointer))
//! );
//! experiment.run();
//! ```
//!
//! # See Also
//!
//! - [`Runable`] - Core trait defining the optimization loop interface
//! - [`Optimizer`] - Trait for optimization algorithms
//! - [`Searchspace`] - Domain definition for optimization variables
//! - [`Stop`] - Stopping criterion interface

use crate::{
    Accumulator, Problem, aliases::{Accum, Cod, CompShape, Func, Opt, Scp}, checkpointer::Checkpointer, optimizer::Sampler, recorder::Recorder, solution::SolId, stop::Stop,
};

// BASICS
pub mod basics;
pub use basics::{Experiment};
// BATCHED
pub mod batched;
// pub use batched::batchevaluator::{BatchEvaluator, ThrBatchEvaluator};
// pub use batched::batchfidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
// SEQUENTIAL
// pub mod sequential;
// pub use sequential::seqevaluator::{SeqEvaluator, ThrSeqEvaluator, VecThrSeqEvaluator};
// pub use sequential::seqfidevaluator::{
//     FidSeqEvaluator, FidThrSeqEvaluator, PoolFidThrSeqEvaluator,
// };

// #[cfg(feature = "mpi")]
// pub use batched::batchfidevaluator::FidDistBatchEvaluator;
// #[cfg(feature = "mpi")]
// pub use sequential::seqevaluator::DistSeqEvaluator;
// #[cfg(feature = "mpi")]
// pub use sequential::seqfidevaluator::FidDistSeqEvaluator;

pub mod parallel;

/// Pool modes for the [`FuncStatePool`](crate::experiment::basics::FuncStatePool) when retrieving [`FuncState`](crate::FuncState).
/// This prevent overflowing RAM whith large [`FuncState`](crate::FuncState) by allowing to retrieve the state from memory if it is still there,
/// or to load it from disk if it has been evicted.
/// It is only used for multi-fidelity optimization where the state of the objective function has to be tracked.
pub enum PoolMode {
    /// Retrieve the state saved in memory (if any) without loading from disk.
    InMemory,
    /// Load the state from disk using the checkpointer.
    Persistent,
}

/// The core trait defining the optimization loop interface.
///
/// [`Runable`] orchestrates the complete optimization pipeline by combining:
/// - [`Searchspace`] - Defines the variable domains
/// - [`Optimizer`] - Generates candidate solutions
/// - [`FuncWrapper`] / Objective function - Evaluates solutions
/// - [`Stop`] - Determines when to terminate
/// - [`Recorder`] - Saves evaluated solutions
/// - [`Checkpointer`] - Saves experiment state for recovery
/// - [`Accumulator`] - Tracks best solutions across the experiment
///
/// # The Optimization Loop
///
/// An implementation of [`Runable`] executes this general loop:
///
/// 1. **Initialization** - Set up optimizer state, load from checkpoint if resuming
/// 2. **Generation** - Optimizer produces candidate solution(s) while updating its
///    internal state based on results, informed by the [`Accumulator`]
/// 3. **Evaluation** - Objective function evaluates solution(s)
/// 4. **Accumulation** - [`update_accumulator`](Runable::update_accumulator) update best solutions for each
///    newly evaluated solution into the running accumulator
/// 5. **Recording** - Save evaluated solutions (if recorder present)
/// 6. **Checkpointing** - Save experiment state (if checkpointer present)
/// 7. **Check Stop** - Evaluate stopping criterion
/// 8. **Repeat** - Continue from step 2 until stop criterion met
///
/// # Accumulator
///
/// The [`Accumulator`] tracks the most relevant solutions evaluated so far.
///
/// - **Single-objective** codomains (implementing [`Single`](crate::Single)) use
///   [`BestAccumulator`](crate::BestAccumulator): the single solution with the highest
///   [`TypeCodom`](crate::Codomain::TypeCodom) value (via [`Ord`]) is retained.
/// - **Multi-objective** codomains (implementing [`Multi`](crate::Multi)) use
///   [`ParetoAccumulator`](crate::ParetoAccumulator): the current Pareto front is maintained
///   using the [`Dominate`](crate::Dominate) relation.
///
/// # Design Philosophy
///
/// The trait separates concerns:
/// - **Optimizer** defines the search strategy (single iteration)
/// - **Runable** defines the execution context (loop, parallelization, I/O)
///
/// This allows:
/// - The same optimizer to run in different contexts (mono/threaded/distributed)
/// - Transparent checkpointing and accumulation without algorithm modifications
/// - Flexible evaluation strategies (batch vs sequential, parallel vs serial)
///
/// # Implementations
///
/// [`Runable`] is implemented for combinations of:
/// - Experiment types: [`MonoExperiment`], [`ThrExperiment`], [`MPIExperiment`]
/// - Optimizer types: [`BatchOptimizer`](crate::optimizer::BatchOptimizer), [`SingleOptimizer`](crate::optimizer::SingleOptimizer)
/// - Objective types: [`Objective`](crate::objective::Objective), [`Stepped`](crate::objective::Stepped)
///
/// # Example: Basic Usage
///
/// ```rust,ignore
/// use tantale::core::{experiment::mono, Objective, stop::Calls};
/// use tantale::algos::RandomSearch;
///
/// // Create experiment
/// let experiment = mono(
///     searchspace,
///     Objective::new(my_function),
///     RandomSearch::new(),
///     Calls::new(100),
///     (None, None)
/// );
///
/// // Run optimization loop
/// experiment.run();
/// ```
///
/// # Example: With Checkpointing
///
/// ```rust,ignore
/// use tantale::core::{MessagePack, FolderConfig, load};
///
/// // Initial run
/// let config = FolderConfig::new("results").init();
/// let checkpointer = MessagePack::new(config);
/// let exp = mono(space, obj, opt, stop, (None, Some(checkpointer)));
/// exp.run();
///
/// // Resume from checkpoint
/// let config = FolderConfig::new("results").init();
/// let checkpointer = MessagePack::new(config).unwrap();
/// let mut exp = load!(mono, MyOpt, MyStop, space, obj, (None, checkpointer));
///
/// // Extend stopping criterion and continue
/// exp.get_mut_stop().add(100);
/// exp.run();
/// ```
///
/// # Type Parameters
///
/// The trait is generic over all experiment components:
/// - `PSol` - The uncomputed solution type (e.g. [`BaseSol`](crate::BaseSol))
/// - `SolId` - Solution identifier type (e.g. [`SId`](crate::SId))
/// - `Scp` - Searchspace type (e.g. [`Sp`](crate::Sp))
/// - `Op` - Optimizer type
/// - `St` - Stopping criterion type (e.g. [`Calls`](crate::stop::Calls))
/// - `Rec` - Recorder type (e.g. [`CSVRecorder`](crate::recorder::CSVRecorder))
/// - `Check` - Checkpointer type (e.g. [`MessagePack`](crate::checkpointer::MessagePack))
/// - `Out` - Outcome type from objective function
/// - `Fn` - Wrapper for the objective function
///
/// # See Also
///
/// - [`mono`], [`threaded`], [`distributed`] - Helper functions to create experiments
/// - `load!` - Macro for loading experiments from checkpoints
/// - [`Optimizer`] - Trait for optimization algorithms
/// - [`Stop`] - Trait for stopping criteria
/// - [`crate::domain::codomain::Accumulator`] - Trait for accumulating best solutions across iterations
/// - [`crate::domain::codomain::BestAccumulator`] - Accumulator for single-objective codomains
/// - [`crate::domain::codomain::ParetoAccumulator`] - Accumulator for multi-objective codomains
pub trait Runable: Sized {
    type SolId: SolId;
    type Problem: Problem<Id = Self::SolId>;
    type Optimizer: Sampler<Self::Problem, Opt<Self::Problem>, Info = <Self::Problem as Problem>::Info>;
    type Stop: Stop;
    type Recorder: Recorder<Self::Problem, Self::Context>;
    type Checkpointer: Checkpointer<Self::Problem, Self::Context>;
    type Context;

    /// Runs the optimization loop to completion.
    ///
    /// Executes the full pipeline, generation, evaluation, accumulation, recording,
    /// checkpointing, and stop-criterion checks, until the [`Stop`] criterion is met.
    /// Consumes `self` so that a finished experiment cannot be accidentally re-run.
    ///
    /// It return an [`Accumulator`] containing the best solutions found during the run.
    fn run(self) -> Accum<Self::Problem>;

    /// Returns a reference to the [`Stop`] criterion.
    fn get_stop(&self) -> &Self::Stop;

    /// Returns a reference to the [`Searchspace`].
    fn searchspace(&self) -> &Scp<Self::Problem>;

    /// Returns a reference to the [`Codomain`](crate::Codomain).
    fn codomain(&self) -> &Cod<Self::Problem>;

    /// Returns a reference to the objective function wrapper.
    fn objective(&self) -> &Func<Self::Problem>;

    /// Returns a reference to the [`Optimizer`].
    fn optimizer(&self) -> &Self::Optimizer;

    /// Returns a reference to the [`Recorder`], if any.
    fn recorder(&self) -> Option<&Self::Recorder>;

    /// Returns a reference to the [`Checkpointer`], if any.
    fn checkpointer(&self) -> Option<&Self::Checkpointer>;

    /// Returns a mutable reference to the [`Stop`] criterion.
    ///
    /// Useful for extending or adjusting the budget after loading from a checkpoint:
    ///
    /// ```rust,ignore
    /// let mut exp = load!(mono, MyOpt, Calls, space, obj, (None, checkpointer));
    /// exp.get_mut_stop().add(500); // run 500 more evaluations
    /// exp.run();
    /// ```
    fn mut_stop(&mut self) -> &mut Self::Stop;

    /// Returns a mutable reference to the [`Searchspace`].
    fn mut_searchspace(&mut self) -> &mut Scp<Self::Problem>;

    /// Returns a mutable reference to the [`Codomain`](crate::Codomain).
    fn mut_codomain(&mut self) -> &mut Cod<Self::Problem>;

    /// Returns a mutable reference to the objective function wrapper.
    fn mut_objective(&mut self) -> &mut Func<Self::Problem>;

    /// Returns a mutable reference to the [`Optimizer`].
    fn mut_optimizer(&mut self) -> &mut Self::Optimizer;

    /// Returns a mutable reference to the [`Recorder`], if any.
    fn mut_recorder(&mut self) -> Option<&mut Self::Recorder>;

    /// Returns a mutable reference to the [`Checkpointer`], if any.
    fn mut_checkpointer(&mut self) -> Option<&mut Self::Checkpointer>;

    /// Returns a read-only reference to the [`Accumulator`].
    fn accumulator(&self) -> &Accum<Self::Problem>;

    /// Returns a mutable reference to the [`Accumulator`].
    fn mut_accumulator(&mut self) -> &mut Accum<Self::Problem>;

    /// Integrates a newly evaluated solution into the [`Accumulator`].
    fn update_accumulator(&mut self, comp: &CompShape<Self::Problem>) {
        self.mut_accumulator().accumulate(comp);
    }
}