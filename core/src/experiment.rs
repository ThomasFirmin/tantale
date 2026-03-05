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
//!     (searchspace, codomain),
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
//!     (searchspace, codomain),
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
//!     (searchspace, codomain),
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
//! ## Sequential Optimization ([`SequentialOptimizer`](crate::optimizer::SequentialOptimizer))
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
//! See the [`objective!`] procedural macro for easy definition.
//! ```rust,ignore
//!
//! // This is automatically generated and wrapped with the objective! procedural macro.
//! fn my_objective(x: &MySolution) -> MyOutcome {
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
//! See the [`objective!`] procedural macro for easy definition.
//!
//! ```rust,ignore
//! // This is automatically generated and wrapped with the objective! procedural macro.
//! fn training_step(x: &Solution, state: TrainingState) -> (TrainingOutcome, TrainingState) {
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
//!     (searchspace, codomain),
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
    SId, Searchspace,
    checkpointer::{Checkpointer, MonoCheckpointer, ThrCheckpointer},
    domain::onto::LinkOpt,
    objective::{FuncWrapper, Outcome},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::CompShape,
    solution::{Batch, Id, OutBatch, SolutionShape, Uncomputed, shape::RawObj},
    stop::Stop,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::DistCheckpointer,
    experiment::mpi::{
        utils::{MPIProcess, SendRec, XMsg},
        worker::Worker,
    },
    solution::{HasY, shape::SolObj},
};
#[cfg(feature = "mpi")]
use ::mpi::Rank;

// BASICS
pub mod basics;
pub use basics::{MonoExperiment, ThrExperiment};
// BATCHED
pub mod batched;
pub use batched::batchevaluator::{BatchEvaluator, ThrBatchEvaluator};
pub use batched::batchfidevaluator::{FidBatchEvaluator, FidThrBatchEvaluator};
// SEQUENTIAL
pub mod sequential;
pub use sequential::seqevaluator::{SeqEvaluator, ThrSeqEvaluator, VecThrSeqEvaluator};
pub use sequential::seqfidevaluator::{
    FidSeqEvaluator, FidThrSeqEvaluator, PoolFidThrSeqEvaluator,
};

#[cfg(feature = "mpi")]
pub use batched::batchfidevaluator::FidDistBatchEvaluator;

#[cfg(feature = "mpi")]
pub use sequential::seqevaluator::DistSeqEvaluator;
#[cfg(feature = "mpi")]
pub use sequential::seqfidevaluator::FidDistSeqEvaluator;

#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use basics::MPIExperiment;

/// Creates a single-threaded experiment ([`MonoExperiment`]).
///
/// This is the primary entry point for creating experiments that execute in a single thread.
/// The experiment combines all necessary components and can be immediately run or inspected.
///
/// # Parameters
///
/// * `space` - A tuple of ([`Searchspace`], [`Codomain`](crate::Codomain)), $f: X \rightarrow Y$
/// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped), $f$
/// * `optimizer` - An optimizer implementing [`Optimizer`]
/// * `stop` - A stopping criterion implementing [`Stop`]
/// * `saver` - A tuple of (optional [`Recorder`], optional [`Checkpointer`])
///
/// # Returns
///
/// An experiment implementing [`Runable`] ready to be executed
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::mono, Objective, stop::Calls};
/// use tantale::algos::{random_search, RandomSearch};
///
/// let sp = my_searchspace();
/// let cod = random_search::codomain(|out| out.value);
/// let obj = Objective::new(my_function);
/// let opt = RandomSearch::new();
/// let stop = Calls::new(100);
///
/// let experiment = mono((sp, cod), obj, opt, stop, (None, None)); // No recording or checkpointing
/// experiment.run();
/// ```
///
/// # See Also
///
/// * [`threaded`] - For multi-threaded execution
/// * [`distributed`] - For MPI-based distributed execution
/// * [`MonoExperiment`] - The underlying experiment type
pub fn mono<PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MonoExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
        space, objective, optimizer, stop, saver,
    )
}

/// Creates a multi-threaded experiment ([`ThrExperiment`]).
///
/// This function creates experiments that execute using multiple OS threads, enabling
/// parallel evaluation of solutions. Particularly effective for batch optimizers or
/// sequential optimizers with varying evaluation times.
///
/// # Parameters
///
/// * `space` - A tuple of ([`Searchspace`], [`Codomain`](crate::Codomain))
/// * `objective` - The objective function wrapped in [`Objective`](crate::objective::Objective) or [`Stepped`](crate::objective::Stepped)
/// * `optimizer` - An optimizer implementing [`Optimizer`]
/// * `stop` - A stopping criterion implementing [`Stop`]
/// * `saver` - A tuple of (optional [`Recorder`], optional [`ThrCheckpointer`])
///
/// # Thread Safety
///
/// All components must implement `Send + Sync` for thread-safe sharing across threads.
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::threaded, Objective, stop::Calls};
/// use tantale::algos::{random_search, BatchRandomSearch};
///
/// let sp = my_searchspace();
/// let cod = random_search::codomain(|out| out.value);
/// let obj = Objective::new(my_function);
/// let opt = BatchRandomSearch::new(10);  // Batch size 10
/// let stop = Calls::new(100);
///
/// let experiment = threaded((sp, cod), obj, opt, stop, (None, None)); // No recording or checkpointing
/// experiment.run();  // Evaluates batches in parallel
/// ```
///
/// # See Also
///
/// * [`mono`] - For single-threaded execution
/// * [`distributed`] - For MPI-based distributed execution
/// * [`ThrExperiment`] - The underlying experiment type
pub fn threaded<PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    ThrExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::new(
        space, objective, optimizer, stop, saver,
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Returns a [`MPIExperiment`].
pub fn distributed<'a, PSol, Scp, Op, St, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: (Scp, Op::Cod),
    objective: Fn,
    optimizer: Op,
    stop: St,
    saver: (Option<Rec>, Option<Check>),
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::new(
        proc,
        space, objective, optimizer, stop, saver
    )
}

/// Loads a single-threaded experiment from a checkpoint.
///
/// Restores a [`MonoExperiment`] from saved state, allowing optimization to be resumed
/// or inspected. All optimizer state, stopping criterion progress, and relevant data
/// are recovered from the checkpoint.
///
/// # Parameters
///
/// * `space` - Tuple of ([`Searchspace`], [`Codomain`](crate::Codomain))
/// * `objective` - The objective function wrapper  
/// * `saver` - Tuple of (optional [`Recorder`], **required** [`Checkpointer`])
///
/// # Type Parameters
///
/// Unlike [`mono`], you must explicitly specify the optimizer and stop types:
/// - `Op` - The optimizer type
/// - `St` - The stopping criterion type
///
/// # Returns
///
/// A loaded experiment implementing [`Runable`], ready to continue or be inspected
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::mono_load, MessagePack, FolderConfig};
/// use tantale::algos::RandomSearch;
/// use tantale::core::stop::Calls;
///
/// // Load from checkpoint
/// let config = FolderConfig::new("my_experiment").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let mut exp = mono_load::<RandomSearch, Calls, _, _, _, _, _, _, _>(
///     (searchspace, codomain),
///     objective,
///     (None, checkpointer)
/// );
///
/// // Inspect or modify before continuing
/// println!("Completed {} evaluations", exp.get_stop().calls());
/// exp.get_mut_stop().add(100);  // Add 100 more evaluations
///
/// // Continue optimization
/// exp.run();
/// ```
///
/// # Note
///
/// Prefer using the [`load!`] macro for cleaner syntax and better type inference.
///
/// # See Also
///
/// * [`mono`] - For creating new experiments
/// * [`threaded_load`] - For loading multi-threaded experiments
/// * [`load!`] - Macro for simpler loading syntax
pub fn mono_load<Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MonoExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, Out>: SolutionShape<SId, Op::SInfo>,
    Rec: Recorder,
    Check: MonoCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MonoExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
        space, objective, saver,
    )
}

/// Loads a multi-threaded experiment from a checkpoint.
///
/// Restores a [`ThrExperiment`] from saved state. The checkpoint must have been created
/// with a thread-safe checkpointer ([`ThrCheckpointer`]).
///
/// # Parameters
///
/// * `space` - Tuple of ([`Searchspace`], [`Codomain`](crate::Codomain))
/// * `objective` - The objective function wrapper
/// * `saver` - Tuple of (optional [`Recorder`], **required** [`ThrCheckpointer`])
///
/// # Type Parameters
///
/// - `Op` - The optimizer type
/// - `St` - The stopping criterion type
///
/// # Example
///
/// ```rust,ignore
/// use tantale::core::{experiment::threaded_load, MessagePack, FolderConfig};
/// use tantale::algos::BatchRandomSearch;
/// use tantale::core::stop::Calls;
///
/// let config = FolderConfig::new("my_experiment").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let mut exp = threaded_load::<BatchRandomSearch, Calls, _, _, _, _, _, _, _>(
///     (searchspace, codomain),
///     objective,
///     (None, checkpointer)
/// );
///
/// exp.run();
/// ```
///
/// # Note
///
/// Prefer using the [`load!`] macro for cleaner syntax.
///
/// # See Also
///
/// * [`threaded`] - For creating new multi-threaded experiments
/// * [`mono_load`] - For loading single-threaded experiments
/// * [`load!`] - Macro for simpler loading syntax
pub fn threaded_load<Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    space: (Scp, Op::Cod),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> impl Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    ThrExperiment<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        Runable<PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<
        Scp,
        PSol,
        SId,
        Op::SInfo,
        <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod,
        Out,
    >: SolutionShape<SId, Op::SInfo>,
    Rec: Recorder,
    Check: ThrCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <ThrExperiment<_, _, _, _, _, _, _, _, _, _> as Runable<_, _, _, _, _, _, _, _, _>>::load(
        space, objective, saver,
    )
}

#[cfg(feature = "mpi")]
#[allow(clippy::type_complexity)]
/// Load a [`MPIExperiment`] from a saver (dist-recorder optional, dist-checkpointer required).
pub fn distributed_load<'a, Op, St, PSol, Scp, Rec, Check, Out, Fn, Eval>(
    proc: &'a MPIProcess,
    space: (
        Scp,
        <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod,
    ),
    objective: Fn,
    saver: (Option<Rec>, Check),
) -> MasterWorker<
    'a,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>,
    PSol,
    SId,
    Scp,
    Op,
    St,
    Rec,
    Check,
    Out,
    Fn,
>
where
    Op: Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>,
    St: Stop,
    PSol: Uncomputed<SId, Scp::Opt, Op::SInfo>,
    MPIExperiment<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn, Eval>:
        MPIRunable<'a, PSol, SId, Scp, Op, St, Rec, Check, Out, Fn>,
    Scp: Searchspace<PSol, SId, Op::SInfo>,
    CompShape<
        Scp,
        PSol,
        SId,
        Op::SInfo,
        <Op as Optimizer<PSol, SId, LinkOpt<Scp>, Out, Scp>>::Cod,
        Out,
    >: SolutionShape<SId, Op::SInfo>,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SId, Op::SInfo>>,
    Eval: Evaluate,
{
    <MPIExperiment<'a, _, _, _, _, _, _, _, _, _, _> as MPIRunable<'a, _, _, _, _, _, _, _, _, _>>::load(
        proc,
        space,
        objective,
        saver
    )
}

/// Macro for loading experiments from checkpoints with simplified syntax.
///
/// This macro provides a convenient interface for calling the `*_load` functions
/// while requiring only the optimizer and stop criterion types to be specified.
/// All other generic types are inferred from the provided arguments.
///
/// # Syntax
///
/// ```text
/// load!(mono, OptimizerType, StopType, space, objective, saver)
/// load!(threaded, OptimizerType, StopType, space, objective, saver)
/// load!(distributed, mpi_proc, OptimizerType, StopType, space, objective, saver)  // MPI feature only
/// ```
///
/// # Parameters
///
/// * First argument: Execution context (`mono`, `threaded`, or `distributed`)
/// * For distributed: `mpi_proc` - Reference to [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess)
/// * `OptimizerType` - The concrete optimizer type (e.g., `RandomSearch`)
/// * `StopType` - The concrete stopping criterion type (e.g., `Calls`)
/// * `space` - Tuple of (searchspace, codomain)
/// * `objective` - The objective function wrapper
/// * `saver` - Tuple of (optional recorder, required checkpointer)
///
/// # Examples
///
/// ## Basic Loading
///
/// ```rust,ignore
/// use tantale::core::{load, MessagePack, FolderConfig};
/// use tantale::algos::RandomSearch;
/// use tantale::core::stop::Calls;
///
/// let config = FolderConfig::new("results").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// let exp = load!(
///     mono,                    // Execution context
///     RandomSearch,            // Optimizer type
///     Calls,                   // Stop criterion type
///     (searchspace, codomain), // Space
///     objective,               // Objective function
///     (None, checkpointer)     // Saver (no recorder, with checkpointer)
/// );
///
/// exp.run();
/// ```
///
/// ## Extending Optimization Budget
///
/// ```rust,ignore
/// let mut exp = load!(mono, MyOpt, Calls, (searchspace, codomain), obj, (None, checkpointer));
///
/// let config = FolderConfig::new("results[`FuncWrapper``] / ").init();
/// let checkpointer = MessagePack::new(config).unwrap();
///
/// // Inspect current progress
/// println!("Completed: {}", exp.get_stop().calls());
///
/// // Add more evaluations
/// exp.get_mut_stop().add(100);
///
/// // Continue optimization
/// exp.run();
/// ```
///
/// ## Multi-Threaded Loading
///
/// ```rust,ignore
/// let exp = load!(threaded, BatchRandomSearch, Calls, space, obj, saver);
/// exp.run();
/// ```
///
/// ## Distributed Loading (MPI feature)
///
/// ```rust,ignore
/// #[cfg(feature = "mpi")]
/// {
///     use tantale::core::{load, MessagePack, FolderConfig, experiment::mpi::utils::MPIProcess};
///     use tantale::algos::RandomSearch;
///     use tantale::core::stop::Calls;
///     
///     let mpi_process = MPIProcess::new();
///
///     let config = FolderConfig::new("results").init(&mpi_process);
///     let checkpointer = MessagePack::new(config).unwrap();
///
///     let exp = load!(
///         distributed,
///         &mpi_process,
///         MyOptimizer,
///         Calls,
///         (searchspace, codomain),
///         obj,
///         (None, checkpointer)
///     );
///     exp.run();
/// }
/// ```
///
/// # See Also
///
/// * [`mono_load`] - Direct function for single-threaded loading
/// * [`threaded_load`] - Direct function for multi-threaded loading  
/// * [`distributed_load`] - Direct function for distributed loading (MPI feature)
/// * [`mono`], [`threaded`], [`distributed`] - For creating new experiments
#[macro_export]
#[cfg(not(feature = "mpi"))]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<$Op, $St>($space, $objective, $saver)/// 3. **Update** -
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<$Op, $St>($space, $objective, $saver)
    };
    (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load::<$Op, $St>($proc, $space, $objective, $saver)
    };
}

/// Macro for loading experiments from checkpoints with simplified syntax (MPI-enabled version).
///
/// This is the MPI-enabled version of the [`load!`] macro. It provides the same functionality
/// as the non-MPI version but with additional support for distributed experiments.
///
/// # See Also
///
/// See the non-MPI [`load!`] macro documentation for detailed usage and examples.
/// The only difference is that this version supports the `distributed` variant.
#[macro_export]
#[cfg(feature = "mpi")]
macro_rules! load {
    (mono, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::mono_load::<$Op, $St, _, _, _, _, _, _, _>($space, $objective, $saver)
    };
    (threaded, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::threaded_load::<$Op, $St, _, _, _, _, _, _, _>(
            $space, $objective, $saver,
        )
    };
    (distributed, $proc:expr, $Op:ty, $St:ty, $space:expr, $objective:expr, $saver:expr) => {
        $crate::experiment::distributed_load::<$Op, $St, _, _, _, _, _, _, _>(
            $proc, $space, $objective, $saver,
        )
    };
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
///
/// # The Optimization Loop
///
/// An implementation of [`Runable`] executes this general loop:
///
/// 1. **Initialization** - Set up optimizer state, load from checkpoint if resuming
/// 2. **Generation** - Optimizer produces candidate solution(s) while updating its internal state based on results
/// 3. **Evaluation** - Objective function evaluates solution(s)
/// 4. **Recording** - Save evaluated solutions (if recorder present)
/// 5. **Checkpointing** - Save experiment state (if checkpointer present)
/// 6. **Check Stop** - Evaluate stopping criterion
/// 7. **Repeat** - Continue from step 2 until stop criterion met
///
/// # Design Philosophy
///
/// The trait separates concerns:
/// - **Optimizer** defines the search strategy (single iteration)
/// - **Runable** defines the execution context (loop, parallelization, I/O)
///
/// This allows:
/// - The same optimizer to run in different contexts (mono/threaded/distributed)
/// - Transparent checkpointing without algorithm modifications
/// - Flexible evaluation strategies (batch vs sequential, parallel vs serial)
///
/// # Implementations
///
/// [`Runable`] is implemented for combinations of:
/// - Experiment types: [`MonoExperiment`], [`ThrExperiment`], [`MPIExperiment`]
/// - Optimizer types: [`BatchOptimizer`](crate::optimizer::BatchOptimizer), [`SequentialOptimizer`](crate::optimizer::SequentialOptimizer)
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
///     (searchspace, codomain),
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
/// - `PSol` - The uncomputed solution type (e.g. [`BasePartial`](crate::BasePartial))
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
/// - [`load!`] - Macro for loading experiments from checkpoints
/// - [`Optimizer`] - Trait for optimization algorithms
/// - [`Stop`] - Trait for stopping criteria
pub trait Runable<PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: Checkpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
{
    fn new(
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> Self;
    fn run(self);
    fn load(space: (Scp, Op::Cod), objective: Fn, saver: (Option<Rec>, Check)) -> Self;
    fn get_stop(&self) -> &St;
    fn get_searchspace(&self) -> &Scp;
    fn get_codomain(&self) -> &Op::Cod;
    fn get_objective(&self) -> &Fn;
    fn get_optimizer(&self) -> &Op;
    fn get_recorder(&self) -> Option<&Rec>;
    fn get_checkpointer(&self) -> Option<&Check>;
    fn get_mut_stop(&mut self) -> &mut St;
    fn get_mut_searchspace(&mut self) -> &mut Scp;
    fn get_mut_codomain(&mut self) -> &mut Op::Cod;
    fn get_mut_objective(&mut self) -> &mut Fn;
    fn get_mut_optimizer(&mut self) -> &mut Op;
    fn get_mut_recorder(&mut self) -> Option<&mut Rec>;
    fn get_mut_checkpointer(&mut self) -> Option<&mut Check>;
}

#[cfg(feature = "mpi")]
pub enum MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    Master(DRun),
    Worker(DRun::WType),
}

#[cfg(feature = "mpi")]
impl<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
    MasterWorker<'a, DRun, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    DRun: MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    pub fn run(self) {
        match self {
            MasterWorker::Master(exp) => exp.run(),
            MasterWorker::Worker(worker) => worker.run(),
        }
    }
}

#[cfg(feature = "mpi")]
/// [`MPIRunable`] describes a MPI-distributed [`Runable`], defined by a [`MasterWorker`] parallelization.
pub trait MPIRunable<'a, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>
where
    Self: Sized,
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    St: Stop,
    Rec: Recorder,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Check: DistCheckpointer,
    Out: Outcome,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    type WType: Worker<SolId>;
    fn new(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        optimizer: Op,
        stop: St,
        saver: (Option<Rec>, Option<Check>),
    ) -> MasterWorker<'a, Self, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
    fn run(self);
    fn load(
        proc: &'a MPIProcess,
        space: (Scp, Op::Cod),
        objective: Fn,
        saver: (Option<Rec>, Check),
    ) -> MasterWorker<'a, Self, PSol, SolId, Scp, Op, St, Rec, Check, Out, Fn>;
    fn get_stop(&self) -> &St;
    fn get_searchspace(&self) -> &Scp;
    fn get_codomain(&self) -> &Op::Cod;
    fn get_objective(&self) -> &Fn;
    fn get_optimizer(&self) -> &Op;
    fn get_recorder(&self) -> Option<&Rec>;
    fn get_checkpointer(&self) -> Option<&Check>;
    fn get_mut_stop(&mut self) -> &mut St;
    fn get_mut_searchspace(&mut self) -> &mut Scp;
    fn get_mut_codomain(&mut self) -> &mut Op::Cod;
    fn get_mut_objective(&mut self) -> &mut Fn;
    fn get_mut_optimizer(&mut self) -> &mut Op;
    fn get_mut_recorder(&mut self) -> Option<&mut Rec>;
    fn get_mut_checkpointer(&mut self) -> Option<&mut Check>;
}

//------------------------//
//-------EVALUATOR--------//
//------------------------//

/// Output type for batch evaluation by [Evaluate].
///
/// Returns a tuple of:
/// - A [`Batch`] of computed solutions with their full shape information
/// - An [`OutBatch`] containing the raw outcomes and IDs
///
/// This type is used by [`BatchEvaluator`] to return evaluated batches.
pub type OutBatchEvaluate<SolId, SInfo, Info, Scp, PSol, Cod, Out> = (
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>,
    OutBatch<SolId, Info, Out>,
);

/// Output type for single solution evaluation.
///
/// Returns a tuple of:
/// - A fully computed solution with complete shape information
/// - A tuple of (solution ID, raw outcome)
///
/// This type is used by sequential evaluators to return individual evaluated solutions.
pub type OutShapeEvaluate<SolId, SInfo, Scp, PSol, Cod, Out> =
    (CompShape<Scp, PSol, SolId, SInfo, Cod, Out>, (SolId, Out));

/// Output type for distributed evaluation with MPI.
///
/// Returns a tuple of:
/// - The MPI rank that performed the evaluation
/// - The evaluated solution and outcome (same structure as [`OutShapeEvaluate`])
///
/// This allows the master process to track which worker evaluated each solution.
#[cfg(feature = "mpi")]
pub type DistOutShapeEvaluate<SolId, SInfo, Scp, PSol, Cod, Out> = (
    Rank,
    (CompShape<Scp, PSol, SolId, SInfo, Cod, Out>, (SolId, Out)),
);

/// Marker trait for evaluation strategies.
///
/// [`Evaluate`] serves as a base trait for all evaluator types in Tantale. Evaluators
/// are responsible for taking uncomputed solutions generated by an optimizer and
/// evaluating them using the objective function.
///
/// # Role in the Pipeline
///
/// Evaluators sit between the optimizer and the objective function:
/// 1. Optimizer generates [`Uncomputed`] solutions
/// 2. Evaluator orchestrates their evaluation (sequencing, parallelization)
/// 3. Objective function computes outcomes
/// 4. Evaluator returns [`Computed`](crate::Computed) solutions to the optimizer
///
/// # Evaluation Strategies
///
/// Three concrete trait extensions define different evaluation strategies:
/// - [`MonoEvaluate`] - Single-threaded sequential evaluation
/// - [`ThrEvaluate`] - Multi-threaded parallel evaluation
/// - [`DistEvaluate`] - MPI-distributed evaluation
///
/// # Implementation Note
///
/// This trait requires `Serialize + Deserialize` to support checkpointing of evaluator state.
pub trait Evaluate
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// Single-threaded evaluation strategy.
///
/// [`MonoEvaluate`] defines how to evaluate solutions in a single thread. This is used
/// by [`MonoExperiment`] to sequentially evaluate each solution without parallelization.
///
/// # Evaluation Flow
///
/// 1. [`init`](Self::init) - Initialize evaluator state
/// 2. Loop:
///    - Optimizer generates uncomputed solution(s)
///    - [`evaluate`](Self::evaluate) - Evaluate using objective function
///    - Return computed solutions with outcomes
///
/// # Implementation Examples
///
/// Concrete implementations include:
/// - [`BatchEvaluator`] - Evaluates batches sequentially
/// - [`SeqEvaluator`] - Evaluates single solutions
/// - [`FidBatchEvaluator`] - Evaluates multi-fidelity batches
/// - [`FidSeqEvaluator`] - Evaluates multi-fidelity solutions
///
/// # Type Parameters
///
/// * `PSol` - Uncomputed solution type
/// * `SolId` - Solution identifier type
/// * `Op` - Optimizer type
/// * `Scp` - Searchspace type
/// * `Out` - Outcome type
/// * `St` - Stopping criterion type
/// * `Fn` - Objective function wrapper
/// * `OutType` - Return type specific to the evaluator (batch or single solution)
///
/// # See Also
///
/// * [`ThrEvaluate`] - For multi-threaded evaluation
/// * [`DistEvaluate`] - For distributed evaluation
/// * [`MonoExperiment`] - Uses this evaluation strategy
pub trait MonoEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initializes the evaluator state.
    ///
    /// Called once before the optimization loop begins. Can be used to set up
    /// internal data structures or reset state.
    fn init(&mut self);

    /// Evaluates solution(s) using the objective function.
    ///
    /// Takes uncomputed solutions from the evaluator's internal state, evaluates them
    /// using the provided objective function and codomain, and returns the results.
    ///
    /// # Parameters
    ///
    /// * `ob` - The objective function wrapper
    /// * `cod` - The codomain for extracting optimization values
    /// * `stop` - Mutable reference to stopping criterion (updated with evaluation progress)
    ///
    /// # Returns
    ///
    /// The evaluated solution(s) in a format specific to the evaluator implementation.
    /// This could be a single solution or a batch depending on the concrete type.
    fn evaluate(&mut self, ob: &Fn, cod: &Op::Cod, stop: &mut St) -> OutType;
}

/// Multi-threaded evaluation strategy.
///
/// [`ThrEvaluate`] defines how to evaluate solutions across multiple threads. This is used
/// by [`ThrExperiment`] to parallelize evaluation, maximizing CPU utilization.
///
/// # Thread Safety
///
/// The objective function and codomain are wrapped in [`Arc`] for safe sharing.
/// The stopping criterion uses `Arc<Mutex<St>>` for synchronized updates
/// across threads.
///
/// # Parallelization Strategies
///
/// - **Batch evaluation**: All solutions in a batch evaluated in parallel
/// - **Sequential on-demand**: Threads request solutions to the optimizer as they complete evaluations
///
/// # Implementation Examples
///
/// Concrete implementations include:
/// - [`ThrBatchEvaluator`] - Parallel batch evaluation
/// - [`VecThrSeqEvaluator`] - Parallel sequential evaluation
/// - [`FidThrBatchEvaluator`] - Parallel multi-fidelity batch evaluation
/// - [`VecFidThrSeqEvaluator`] - Parallel multi-fidelity sequential
///
/// # Type Parameters
///
/// * `PSol` - Uncomputed solution type (must be `Send + Sync`)
/// * `SolId` - Solution identifier type
/// * `Op` - Optimizer type
/// * `Scp` - Searchspace type (must be `Send + Sync`)
/// * `Out` - Outcome type (must be `Send + Sync`)
/// * `St` - Stopping criterion type (must be `Send + Sync`)
/// * `Fn` - Objective function wrapper (must be `Send + Sync`)
/// * `OutType` - Return type specific to the evaluator
///
/// # See Also
///
/// * [`MonoEvaluate`] - For single-threaded evaluation
/// * [`DistEvaluate`] - For distributed evaluation
/// * [`ThrExperiment`] - Uses this evaluation strategy
pub trait ThrEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>: SolutionShape<SolId, Op::SInfo>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
{
    /// Initializes the evaluator state.
    fn init(&mut self);

    /// Evaluates solution(s) using multiple threads.
    ///
    /// Spawns or coordinates worker threads to evaluate solutions in parallel.
    /// The objective function, codomain, and stopping criterion are shared safely
    /// across threads using [`Arc`] and [`Mutex`].
    ///
    /// # Parameters
    ///
    /// * `ob` - Shared objective function wrapper ([`Arc`])
    /// * `cod` - Shared codomain ([`Arc`])
    /// * `stop` - Shared stopping criterion (`Arc<Mutex<St>>`)
    ///
    /// # Returns
    ///
    /// The evaluated solutions after parallel evaluation completes.
    fn evaluate(&mut self, ob: Arc<Fn>, cod: Arc<Op::Cod>, stop: Arc<Mutex<St>>) -> OutType;
}

#[cfg(feature = "mpi")]
/// Distributed evaluation strategy using MPI.
///
/// [`DistEvaluate`] defines how to evaluate solutions across multiple processes using
/// MPI (Message Passing Interface). This is used by [`MPIExperiment`] to distribute
/// evaluation across a cluster or multi-node system.
///
/// # MPI Architecture
///
/// - **Master (rank 0)**: Runs optimizer, distributes work, collects results
/// - **Workers (rank > 0)**: Wait for solutions, evaluate, return outcomes
///
/// # Communication
///
/// The [`SendRec`] parameter handles MPI message passing:
/// - Sending solutions from master to workers
/// - Receiving evaluated solutions from workers back to master
/// - Message types defined by `M: XMsg` trait
///
/// # Implementation Examples
///
/// Concrete implementations include:
/// - [`BatchEvaluator`] - Distributed batch evaluation
/// - [`DistSeqEvaluator`] - Distributed sequential evaluation
/// - [`FidDistBatchEvaluator`] - Distributed multi-fidelity batch evaluation
/// - [`FidDistSeqEvaluator`] - Distributed multi-fidelity sequential evaluation
///
/// # Type Parameters
///
/// * `PSol` - Uncomputed solution type
/// * `SolId` - Solution identifier type
/// * `Op` - Optimizer type
/// * `Scp` - Searchspace type
/// * `Out` - Outcome type
/// * `St` - Stopping criterion type
/// * `Fn` - Objective function wrapper
/// * `M` - MPI message type implementing [`XMsg`]
/// * `OutType` - Return type specific to the evaluator
///
/// # See Also
///
/// * [`MonoEvaluate`] - For single-threaded evaluation
/// * [`ThrEvaluate`] - For multi-threaded evaluation
/// * [`MPIExperiment`] - Uses this evaluation strategy
/// * [`MPIProcess`] - MPI process management
pub trait DistEvaluate<PSol, SolId, Op, Scp, Out, St, Fn, M, OutType>: Evaluate
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Scp: Searchspace<PSol, SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        SolutionShape<SolId, Op::SInfo> + HasY<Op::Cod, Out>,
    Out: Outcome,
    St: Stop,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
    M: XMsg<SolObj<Scp::SolShape, SolId, Op::SInfo>, SolId, Scp::Obj, Op::SInfo>,
{
    /// Initializes the evaluator state for MPI communication.
    fn init(&mut self);

    /// Evaluates solution(s) using distributed MPI workers.
    ///
    /// The master process uses this method to distribute solutions to worker processes,
    /// coordinate their evaluation, and collect results. This method handles:
    /// - Sending solutions to available workers
    /// - Tracking which workers are busy/idle
    /// - Receiving evaluated solutions as they complete
    /// - Updating the stopping criterion with evaluation progress
    ///
    /// # Parameters
    ///
    /// * `sendrec` - MPI communication handler for sending/receiving solutions
    /// * `ob` - The objective function wrapper (used by workers)
    /// * `cod` - The codomain for extracting optimization values
    /// * `stop` - Mutable reference to stopping criterion
    ///
    /// # Returns
    ///
    /// The evaluated solution(s) collected from worker processes, along with
    /// which rank evaluated each solution (see [`DistOutShapeEvaluate`]).
    fn evaluate(
        &mut self,
        sendrec: &mut SendRec<'_, M, Scp::SolShape, SolId, Op::SInfo, Op::Cod, Out>,
        ob: &Fn,
        cod: &Op::Cod,
        stop: &mut St,
    ) -> OutType;
}
