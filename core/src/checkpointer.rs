//! # Checkpointing
//!
//! This module provides checkpointing capabilities for optimization processes.
//! Checkpointing allows saving and loading the complete state of an
//! optimization experiment, enabling recovery from failures and resumption of long-running optimizations.
//!
//! ## Overview
//!
//! Checkpointing in Tantale is designed to save and restore the state of an optimization experiment at each iteration.
//!
//! ## State Components
//!
//! A checkpoint captures four essential components of an optimization experiment:
//!
//! 1. **[`OptState`]** - The internal state of the [`Optimizer`](crate::optimizer::Optimizer), containing algorithm-specific
//!    information (e.g., population history, learned models, covariance matrices).
//! 2. **[`Stop`]** - The termination criteria state, tracking progress toward stopping conditions (iterations, time elapsed, evaluations).
//! 3. **[`Evaluate`]** - The evaluation context, including cached results and solution tracking (varies by experiment type).
//! 4. **[`GlobalParameters`]** - Global counters for solutions (`sold_id`), optimizer steps (`opt_id`), and experiment runs (`run_id`).
//!
//! ## Checkpoint Variants
//!
//! Tantale provides specialized checkpointer traits for different execution contexts:
//!
//! ### Sequential Experiments
//! - **[`Checkpointer`]** - Basic abstract framework where a single global state is maintained.
//!
//! ### Threaded Experiments  
//! - **[`ThrCheckpointer`]** - For multi-threaded optimization where each worker thread has its own [`Evaluate`] state.
//!   Extends [`Checkpointer`] with thread-local evaluation state management.
//!
//! ### Distributed Experiments (MPI)
//! - **[`DistCheckpointer`]** - For MPI-based distributed optimization coordinated by a master rank.
//! - **[`WorkerCheckpointer`]** - For individual MPI worker processes to checkpoint their local state.
//!
//! ## Usage
//!
//! ```ignore
//! // Scenario 1: Starting a new optimization
//! let mut checkpointer = MessagePack::new(config)?;
//! // ... optimization loop with periodic checkpointing ...
//! checkpointer.save_state(&opt_state, &stop, &eval);
//!
//! // Scenario 2: Resuming from checkpoint
//! let mut checkpointer = MessagePack::new(config)?;
//! let (opt_state, stop, eval) = checkpointer.load()?;
//! // ... continue optimization loop with restored state ...
//! checkpointer.save_state(&opt_state, &stop, &eval);
//! ```
//!
//! ## Implementation Notes
//!
//! - All checkpoint data must implement [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize).
//! - Checkpointers are configuration-dependent via the [`SaverConfig`](crate::SaverConfig) trait.
//! - The [`MessagePack`] implementation is the primary concrete checkpointer for most use cases.
//! - MPI-specific traits are only available with the `mpi` feature.
//! - See [`experiment::Runable`](crate::experiment::Runable) for how checkpointers are integrated into optimization loops.

#[cfg(feature = "mpi")]
use std::panic;

#[cfg(feature = "mpi")]
use crate::experiment::mpi::{utils::MPIProcess, worker::WorkerState};
use crate::{
    Accumulator, Codomain, FuncState, GlobalParameters, HasY, Id, Outcome, SaverConfig, SolInfo,
    SolutionShape, config::NoConfig, experiment::Evaluate, optimizer::OptState, stop::Stop,
};

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod messagepack;
pub use messagepack::MessagePack;
use serde::{Deserialize, Serialize};

pub use crate::errors::CheckpointError;

// Marker trait for checkpointing management in optimization experiments.
///
/// A [`Checkpointer`] provides the ability to save and restore the complete state of an optimization
/// experiment, enabling resumption after interruptions. It manages the persistence of four key state
/// components necessary to restart an experiment from any checkpoint.
///
/// # See Also
/// - [`MonoCheckpointer`] - For single-threaded optimization experiments
/// - [`ThrCheckpointer`] - For multi-threaded optimization experiments
/// - [`DistCheckpointer`] - For distributed MPI-based optimization experiments
pub trait Checkpointer
where
    Self: Sized,
{
    /// Associated type for function state checkpointer, used to manage per-function state.
    type FnStateCheck: FuncStateCheckpointer;
    /// Creates a new function state checkpointer for managing per-function state.
    fn new_func_state_checkpointer(&self) -> Self::FnStateCheck;
}

pub trait FuncStateCheckpointer {
    /// Save the [`FuncState`] for a specific function ID in a threaded optimization experiment.
    fn save_func_state<FnState: FuncState, SolId: Id>(&self, id: &SolId, func_state: &FnState);
    /// Loads all the [`FuncState`] for a specific function ID in a threaded optimization experiment.
    fn load_func_state<FnState: FuncState, SolId: Id>(&self, id: &SolId) -> Option<FnState>;
    /// Removes the [`FuncState`] for a specific function ID in a threaded optimization experiment, returning whether the state was successfully removed.
    fn remove_func_state<SolId: Id>(&self, id: &SolId) -> Result<bool, CheckpointError>;
    /// Load all the [`FuncState`] returning a vector of function states for each function ID.
    fn load_all_func_state<FnState: FuncState, SolId: Id>(&self) -> Vec<(SolId, FnState)>;
}

/// Core trait for checkpointing management in optimization experiments.
///
/// A [`Checkpointer`] provides the ability to save and restore the complete state of an optimization
/// experiment, enabling resumption after interruptions. It manages the persistence of four key state
/// components necessary to restart an experiment from any checkpoint.
///
/// # Trait Purpose
///
/// This trait abstracts checkpoint storage and retrieval, allowing different implementations to use
/// different backends (files, databases, etc.). It requires all state components to be serializable
/// via the [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) traits, typically
/// handled through `serde` derive macros.
///
/// # State Components Managed
///
/// - [`OptState`] - Optimizer Algorithm State
///   Contains the internal data structure of the optimizer. This might include:
///   This state is algorithm-specific and must be implemented for each optimizer.
///
/// - [`Stop`] - Termination Criteria State
///   Tracks progress toward stopping conditions.
///
/// - [`Evaluate`] - Evaluation Context State
///   Maintains evaluation-specific information, varying by experiment type:
///
/// - [`GlobalParameters`] - Global Counters
///   Global parameters of the experiment.
///
/// These counters ensure consistent identification of solutions when resuming.
///
/// # Type Parameters and Associated Types
///
/// ## Associated Type: `Config`
/// - Implements [`SaverConfig`] trait
/// - Provides configuration for checkpoint storage (e.g., file paths)
/// - Used by [`MessagePack`] to determine file locations
/// - Typically [`FolderConfig`](crate::FolderConfig) for file-based checkpointing
///
/// # Relationship to Other Traits
///
/// - **[`ThrCheckpointer`]** - Extends this trait for multi-threaded optimization with per-thread [`Evaluate`] states
/// - **[`DistCheckpointer`]** - Adds distributed (MPI) checkpoint coordination
/// - **[`WorkerCheckpointer`]** - Handles individual MPI worker checkpointing
///
/// # See Also
///
/// - [`experiment::Runable`](crate::experiment::Runable) - Integration point where [`Checkpointer`] is used in optimization loops
/// - [`MessagePack`] - Concrete implementation using MessagePack serialization
/// - [`FolderConfig`](crate::FolderConfig) - Configuration for file-based checkpoint storage
pub trait MonoCheckpointer: Checkpointer {
    /// Configuration type that implements [`SaverConfig`].
    ///
    /// This associated type determines how the checkpointer is configured.
    /// Typically set to [`FolderConfig`](crate::FolderConfig) for file-based checkpointing.
    type Config: SaverConfig;

    /// Initializes a new checkpoint structure for a fresh optimization experiment.
    ///
    /// This method must be called before starting a new optimization that will be checkpointed.
    /// It creates the necessary storage infrastructure (e.g., directories and files) based on the
    /// checkpointer's configuration.
    ///
    /// # Panics
    ///
    /// Should panic if:
    /// - The checkpoint location already exists (preventing accidental overwrite of previous experiments)
    /// - The checkpoint path points to a file instead of a directory
    /// - Directory creation fails due to permission issues or system errors
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = FolderConfig::new(Path::new("./my_experiment"));
    /// let mut checkpointer = MessagePack::new(Arc::new(config))?;
    /// checkpointer.init();  // Creates ./my_experiment/checkpoint/
    /// ```
    ///
    /// # See Also
    /// - [`before_load`](Self::before_load) - Used when resuming an existing experiment
    fn init(&mut self);

    /// Verifies checkpoint integrity before loading a saved experiment.
    ///
    /// This method must be called before using any `load_*` methods.
    /// It validates that:
    /// - The checkpoint directory exists
    /// - All required checkpoint files are present and intact
    /// - The checkpoint structure matches expectations
    ///
    /// # Panics
    ///
    /// Should panic if the checkpoint is invalid or incomplete, indicating data corruption or
    /// incomplete checkpoint writes.
    ///
    /// # See Also
    /// - [`init`](Self::init) - Used when starting a new experiment
    fn before_load(&mut self);
    /// Performs any necessary cleanup or state preparation after loading a checkpoint.
    /// This method can be used to create a new checkpoint directory after loading an existing one.
    /// It should be called after successfully loading a checkpoint and before resuming optimization.
    fn after_load(&mut self);
    /// Saves the complete state of an optimization experiment to a checkpoint.
    ///
    /// This is the primary checkpoint method.
    /// It should be called periodically during optimization to enable recovery from failures.
    ///
    /// # Checkpointed Components
    ///
    /// The method persists:
    /// - [`OptState`] The optimizer's internal state
    /// - [`Stop`] The termination criteria state
    /// - [`Evaluate`] The evaluation context state
    /// - [`GlobalParameters`] Solution, optimizer, run IDs...
    ///
    /// # Example
    ///
    /// ```ignore
    /// checkpointer.save_state(&opt_state, &stop, &eval);
    /// ```
    fn save_state<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
    );

    /// Loads the complete state of an optimization experiment from a checkpoint.
    ///
    /// This is the primary checkpoint recovery method, restoring all three essential state components.
    /// Call [`before_load`](Self::before_load) before using this method to verify checkpoint integrity.
    ///
    /// # Type Parameters
    ///
    /// * `OState` - The optimizer's state type implementing [`OptState`]
    /// * `St` - The stopping criteria type implementing [`Stop`]
    /// * `Eval` - The evaluation state type implementing [`Evaluate`]
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The restored optimizer state
    /// - The restored termination criteria state
    /// - The restored evaluation context state
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError`] if:
    /// - Checkpoint files don't exist
    /// - Deserialization fails (corrupted checkpoint data)
    /// - File I/O errors occur
    ///
    /// # Example
    ///
    /// ```ignore
    /// checkpointer.before_load()?;
    /// let (opt_state, stop, eval) = checkpointer.load()?;
    /// ```
    fn load<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Eval), CheckpointError>;

    /// Loads only the termination criteria state from a checkpoint.
    fn load_stop<St: Stop>(&self) -> Result<St, CheckpointError>;

    /// Loads only the optimizer state from a checkpoint.
    fn load_optimizer<OState: OptState>(&self) -> Result<OState, CheckpointError>;

    /// Loads only the evaluation state from a checkpoint.
    fn load_evaluate<Eval: Evaluate>(&self) -> Result<Eval, CheckpointError>;

    /// Loads the global parameters from a checkpoint.
    fn load_parameters(&self) -> Result<GlobalParameters, CheckpointError>;

    /// Saves the state of an accumulator to a checkpoint.
    fn save_accumulator<Acc, C, SolId, SInfo, Cod, Out>(&self, acc: &Acc)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;

    /// Loads the state of an accumulator from a checkpoint.
    fn load_accumulator<Acc, C, SolId, SInfo, Cod, Out>(&self) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;
}

/// Extended checkpointer trait for multi-threaded optimization experiments.
///
/// A [`ThrCheckpointer`] extends [`Checkpointer`] to support multi-threaded experiments where each worker
/// thread maintains its own [`Evaluate`] state. This is essential for experiments using thread pools where
/// each thread evaluates solutions independently and needs to checkpoint its local progress.
///
/// # Relationship to Checkpointer
///
/// [`ThrCheckpointer`] extends [`Checkpointer`] by adding thread-aware variants of key methods:
/// - `init()` → `init_thr()` - Initialize for multi-threaded storage
/// - `before_load()` → `before_load_thr()` - Verify multi-threaded checkpoint
/// - `save_state()` → `save_state_thr()` - Save with thread context
/// - `load()` → `load_thr()` - Restore multiple thread states
/// - Individual load methods remain unchanged (they apply to shared state)
///
/// # See Also
///
/// - [`MonoCheckpointer`] - For single-threaded optimization
/// - [`DistCheckpointer`] - For distributed MPI optimization
pub trait ThrCheckpointer: Checkpointer {
    /// Configuration type (inherited from [`Checkpointer`]).
    ///
    /// Must implement [`SaverConfig`].
    type Config: SaverConfig;

    /// Equivalent to [`init`](MonoCheckpointer::init) for threaded optimization experiment.
    fn init_thr(&mut self);
    /// Equivalent to [`before_load`](MonoCheckpointer::before_load) for threaded optimization experiment.
    fn before_load_thr(&mut self);
    /// Equivalent to [`after_load`](MonoCheckpointer::after_load) for threaded optimization experiment.
    fn after_load(&mut self);
    /// Equivalent to [`save_state`](MonoCheckpointer::save_state) for threaded optimization experiment.
    fn save_state_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        thread: usize,
    );
    fn load_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Vec<Eval>), CheckpointError>;
    /// Loads all the thread-specific evaluation states from a checkpoint, returning a vector of evaluation states for each thread.
    fn load_all_evaluate_thr<Eval: Evaluate>(&self) -> Result<Vec<Eval>, CheckpointError>;
    /// Equivalent to [`load_stop`](MonoCheckpointer::load_stop) for threaded optimization experiment, returning thread-specific stop state.
    fn load_stop_thr<St: Stop>(&self) -> Result<St, CheckpointError>;
    /// Equivalent to [`load_optimizer`](MonoCheckpointer::load_optimizer) for threaded optimization experiment, returning thread-specific optimizer state.
    fn load_optimizer_thr<OState: OptState>(&self) -> Result<OState, CheckpointError>;
    /// Equivalent to [`load_parameters`](MonoCheckpointer::load_parameters) for threaded optimization experiment, returning shared global parameters.
    fn load_parameters_thr(&self) -> Result<GlobalParameters, CheckpointError>;

    /// Saves the state of an accumulator to a checkpoint.
    fn save_accumulator_thr<Acc, C, SolId, SInfo, Cod, Out>(&self, acc: &Acc)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;

    /// Loads the state of an accumulator from a checkpoint.
    fn load_accumulator_thr<Acc, C, SolId, SInfo, Cod, Out>(&self) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;
}

#[cfg(feature = "mpi")]
/// Checkpointer for individual MPI worker processes in distributed optimization.
///
/// A [`WorkerCheckpointer`] enables checkpointing for individual worker ranks in MPI-based distributed optimization.
/// Each worker process independently checkpoints its local state, allowing recovery if a specific worker fails.
///
/// # Checkpoint Scope
///
/// Each worker checkpoints:
/// - Its own [`WorkerState`] - Algorithm-specific local state for that rank
/// - NO shared optimizer state (handled by master via [`DistCheckpointer`])
/// - NO shared stop criteria (handled by master)
///
/// # Usage
///
/// ```ignore
/// // On master (rank 0)
/// let mut checkpointer = MessagePack::new(config)?;
/// checkpointer.init_dist(&proc);
///
/// // On each worker (ranks 1..N)
/// let mut worker_checkpointer = checkpointer.get_check_worker(&proc);
/// worker_checkpointer.init(&proc);
/// // ... evaluation loop ...
/// worker_checkpointer.save_state(&wasterorker_state, rank);
/// ```
///
/// # See Also
///
/// - [`DistCheckpointer`] - For controller-side distributed checkpointing
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) - MPI context information
/// - [`WorkerState`] - Type trait for worker-local state
pub trait WorkerCheckpointer<WState>
where
    WState: WorkerState,
{
    type FnStateCheck: FuncStateCheckpointer;
    /// Creates a new function state checkpointer for managing per-function state.
    fn new_func_state_checkpointer(&self) -> Self::FnStateCheck;

    /// Initializes checkpoint storage for a worker process in a new distributed experiment.
    ///
    /// This methodaster prepares the checkpoiasternt infrastructure for a specific worasterker rank.
    /// It should be called once during worker initialization.
    ///
    /// # Parameters
    ///
    /// * `proc` - The [`MPIProcess`] context containing rank, comm²unicator, and cluster information
    ///
    ///aster # Panics
    ///
    /// Should panic if:
    /// - Rank-specific checkpoint location already exists (prevents overwriting)
    /// - Directory creation fails
    /// - Checkpoint path is invalid
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut worker_check = checkpointer.get_check_worker::<MyWorkerState>(&proc);
    /// worker_check.init(&proc);
    /// ```
    ///
    /// # See Also
    /// - [`before_load`](Self::before_load) - Used when resuming an existing distributed experiment
    fn init(&mut self, proc: &MPIProcess);

    /// Verifies worker checkpoint integrity before loading an existing distributed experiment.
    ///
    /// This method must be called before calling any `load_*` methods
    /// on this worker. It validates that the checkpoint for the current rank exists and is intact.
    ///
    /// # Parameters
    ///
    /// * `proc` - The [`MPIProcess`] context containing rank and process information
    ///
    /// # Panics
    ///
    /// Should panic if:
    /// - Checkpoint directory or files don't exist
    /// - Checkpoint structure is invalid
    /// - Checkpoint is incomplete or corrupt
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut worker_check = checkpointer.get_check_worker::<MyWorkerState>(&proc);
    /// worker_check.before_load(&proc)?;  // Verify checkpoint exists
    /// let state = worker_check.load(rank)?;
    /// ```
    ///
    /// # See Also
    /// - [`init`](Self::init) - Used when starting a new distributed experiment
    fn before_load(&mut self, proc: &MPIProcess);
    /// Equivalent to [`after_load`](MonoCheckpointer::after_load) for threaded optimization experiment.
    fn after_load(&mut self, proc: &MPIProcess);
    /// Saves the local worker state to a checkpoint.
    ///
    /// This method persists the [`WorkerState`] for a specific MPI rank. It should be called
    /// periodically as the worker makes progress in its evaluation loop.
    ///
    /// # Parameters
    ///
    /// * `state` - The worker's local evaluation and tracking state
    /// * `rank` - The MPI [`Rank`] of this worker (typically matches the rank in `proc`)
    fn save_state(&self, state: &WState, rank: Rank);

    /// Loads the local worker state from a checkpoint.
    ///
    /// This method retrieves the [`WorkerState`] for a specific MPI rank, enabling a worker
    /// to resume from a previous checkpoint.
    ///
    /// # Parameters
    ///
    /// * `rank` - The MPI [`Rank`] whose checkpoint should be loaded
    fn load(&self, rank: Rank) -> Result<WState, CheckpointError>;
}

#[cfg(feature = "mpi")]
/// Distributed checkpoint manager for MPI-based optimization.
///
/// A [`DistCheckpointer`] extends [`Checkpointer`] to manage checkpointing for distributed MPI-based
/// optimization. It coordinates checkpointing across multiple ranks while each maintains independent state.
///
/// # Distributed MPI Architecture
///
/// Tantale's distributed optimization uses a hierarchical architecture:
/// - **Master (typically rank 0)**: Maintains global optimizer state, stop criteria, and global parameters
/// - **Workers (ranks 1..N)**: Evaluate solutions independently and maintain local [`WorkerState`]
///
/// [`DistCheckpointer`] manages the master-side state while delegating worker-level checkpointing
/// to per-rank [`WorkerCheckpointer`] instances.
///
/// # Checkpoint State Distribution
///
/// ```text
/// DistCheckpointer (Master-side)
/// └── Manages:
///     ├── Global optimizer state (OptState)
///     ├── Global stop criteria (Stop)
///     ├── Global parameters (GlobalParameters)
///     └── Provides WorkerCheckpointer for each rank
///
/// WorkerCheckpointer (per-rank)
/// └── Manages:
///     └── Rank-specific worker state (WorkerState)
/// ```
///
/// # Lifecycle Pattern
///
/// ```ignore
/// // On Controller (rank 0)
/// let mut checkpointer = MessagePack::new(config)?;
/// checkpointer.init_dist(&proc);
/// // ... optimization loop ...
/// checkpointer.save_state_dist(&opt_state, &stop, &eval, rank);
///
/// // On each Worker (ranks 1..N)
/// let mut worker_check = checkpointer.get_check_worker(&proc);
/// worker_check.init(&proc);
/// // ... evaluation loop ...
/// worker_check.save_state(&worker_state, rank);
/// ```
///
/// # Relationship to Other Traits
///
/// - **[`WorkerCheckpointer`]** - Worker-side counterpart for per-rank local state
/// - **[`MonoCheckpointer`]** - Single-threaded base trait
/// - **[`ThrCheckpointer`]** - Multi-threaded variant (unrelated to distribution)
///
/// # See Also
///
/// - [`WorkerCheckpointer`] - For worker-local state checkpointing
/// - [`MPIProcess`](crate::experiment::mpi::utils::MPIProcess) - MPI context
pub trait DistCheckpointer: Checkpointer
where
    Self: Sized,
{
    /// Associated worker checkpointer type, specialized for a given [`WorkerState`] type.
    ///
    /// This associated type allows different worker state types to have their own
    /// specialized checkpointer implementations.
    type WCheck<WState: WorkerState>: WorkerCheckpointer<WState, FnStateCheck = Self::FnStateCheck>;

    /// Equivalent to [`init`](MonoCheckpointer::init) for distributed optimization experiments.
    fn init_dist(&mut self, proc: &MPIProcess);
    /// Equivalent to [`before_load`](MonoCheckpointer::before_load) for distributed optimization experiments.
    fn before_load_dist(&mut self, proc: &MPIProcess);
    /// Equivalent to [`after_load`](MonoCheckpointer::after_load) for distributed optimization experiments.
    fn after_load_dist(&mut self, proc: &MPIProcess);
    /// Define an initialization for [`Workers`](crate::experiment::mpi::worker::Worker) that do not have a [`Checkpointer`].
    fn no_check_init(proc: &MPIProcess);
    /// Equivalent to [`save_state`](MonoCheckpointer::save_state) for distributed optimization experiments, with rank context.
    fn save_state_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        rank: Rank,
    );
    /// Equivalent to [`load`](MonoCheckpointer::load) for distributed optimization experiments, returning rank-specific states.
    fn load_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        rank: Rank,
    ) -> Result<(OState, St, Eval), CheckpointError>;
    /// Equivalent to [`load_stop`](MonoCheckpointer::load_stop) for distributed optimization experiments, returning rank-specific stop state.
    fn load_stop_dist<St: Stop>(&self, rank: Rank) -> Result<St, CheckpointError>;
    /// Equivalent to [`load_optimizer`](MonoCheckpointer::load_optimizer) for distributed optimization experiments, returning rank-specific optimizer state.
    fn load_optimizer_dist<OState: OptState>(&self, rank: Rank) -> Result<OState, CheckpointError>;
    /// Equivalent to [`load_evaluate`](MonoCheckpointer::load_evaluate) for distributed optimization experiments, returning rank-specific evaluation state.
    fn load_evaluate_dist<Eval: Evaluate>(&self, rank: Rank) -> Result<Eval, CheckpointError>;
    /// Equivalent to [`load_parameters`](MonoCheckpointer::load_parameters) for distributed optimization experiments, returning global parameters (same across ranks).
    fn load_parameters_dist(&self, rank: Rank) -> Result<GlobalParameters, CheckpointError>;
    /// Saves the state of an accumulator to a checkpoint.
    fn save_accumulator_dist<Acc, C, SolId, SInfo, Cod, Out>(&self, acc: &Acc, rank: Rank)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;

    /// Loads the state of an accumulator from a checkpoint.
    fn load_accumulator_dist<Acc, C, SolId, SInfo, Cod, Out>(
        &self,
        rank: Rank,
    ) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome;
    /// Retrieves the worker checkpointer.
    fn get_check_worker<WState: WorkerState>(&self, proc: &MPIProcess) -> Self::WCheck<WState>;
}

/// An empty [`Checkpointer`] implementation that performs no checkpointing.
#[derive(Serialize, Deserialize)]
pub struct NoCheck;

#[derive(Serialize, Deserialize)]
pub struct NoFuncStateCheck;

impl FuncStateCheckpointer for NoFuncStateCheck {
    fn save_func_state<FnState: FuncState, SolId: Id>(&self, _id: &SolId, _func_state: &FnState) {
        panic!("NoCheck should not be called to save function state.")
    }

    fn load_func_state<FnState: FuncState, SolId: Id>(&self, _id: &SolId) -> Option<FnState> {
        panic!("NoCheck should not be called to load function state.")
    }

    fn remove_func_state<SolId: Id>(&self, _id: &SolId) -> Result<bool, CheckpointError> {
        panic!("NoCheck should not be called to remove function state.")
    }

    fn load_all_func_state<FnState: FuncState, SolId: Id>(&self) -> Vec<(SolId, FnState)> {
        panic!("NoCheck should not be called to load all function state.")
    }
}

impl Checkpointer for NoCheck {
    type FnStateCheck = NoFuncStateCheck;

    fn new_func_state_checkpointer(&self) -> Self::FnStateCheck {
        NoFuncStateCheck
    }
}

#[cfg(feature = "mpi")]
impl MonoCheckpointer for NoCheck {
    type Config = NoConfig;

    fn init(&mut self) {}

    fn before_load(&mut self) {}

    fn save_state<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        _state: &OState,
        _stop: &St,
        _eval: &Eval,
    ) {
    }

    fn after_load(&mut self) {}

    fn load<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Eval), CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_stop<St: Stop>(&self) -> Result<St, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_optimizer<OState: OptState>(&self) -> Result<OState, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_evaluate<Eval: Evaluate>(&self) -> Result<Eval, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_parameters(&self) -> Result<GlobalParameters, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn save_accumulator<Acc, C, SolId, SInfo, Cod, Out>(&self, _acc: &Acc)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to save an accumulator.")
    }

    fn load_accumulator<Acc, C, SolId, SInfo, Cod, Out>(&self) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to load an experiment.")
    }
}

impl ThrCheckpointer for NoCheck {
    type Config = NoConfig;

    fn init_thr(&mut self) {}

    fn before_load_thr(&mut self) {}

    fn after_load(&mut self) {}

    fn save_state_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        _state: &OState,
        _stop: &St,
        _eval: &Eval,
        _thread: usize,
    ) {
        panic!("NoCheck should not be called to save an experiment.")
    }

    fn load_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Vec<Eval>), CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_all_evaluate_thr<Eval: Evaluate>(&self) -> Result<Vec<Eval>, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_stop_thr<St: Stop>(&self) -> Result<St, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_optimizer_thr<OState: OptState>(&self) -> Result<OState, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn load_parameters_thr(&self) -> Result<GlobalParameters, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn save_accumulator_thr<Acc, C, SolId, SInfo, Cod, Out>(&self, _acc: &Acc)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to save an accumulator.")
    }

    fn load_accumulator_thr<Acc, C, SolId, SInfo, Cod, Out>(&self) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to load an experiment.")
    }
}

#[cfg(feature = "mpi")]
impl DistCheckpointer for NoCheck {
    type WCheck<WState: WorkerState> = NoWCheck;

    fn init_dist(&mut self, _proc: &MPIProcess) {}
    fn before_load_dist(&mut self, _proc: &MPIProcess) {}
    fn after_load_dist(&mut self, _proc: &MPIProcess) {}
    fn no_check_init(_proc: &MPIProcess) {}
    fn save_state_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        _state: &OState,
        _stop: &St,
        _eval: &Eval,
        _rank: Rank,
    ) {
    }
    fn load_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        _rank: Rank,
    ) -> Result<(OState, St, Eval), CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
    fn load_stop_dist<St: Stop>(&self, _rank: Rank) -> Result<St, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
    fn load_optimizer_dist<OState: OptState>(
        &self,
        _rank: Rank,
    ) -> Result<OState, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
    fn load_evaluate_dist<Eval: Evaluate>(&self, _rank: Rank) -> Result<Eval, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
    fn load_parameters_dist(&self, _rank: Rank) -> Result<GlobalParameters, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn save_accumulator_dist<Acc, C, SolId, SInfo, Cod, Out>(&self, _acc: &Acc, _rank: Rank)
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to save an accumulator.")
    }

    fn load_accumulator_dist<Acc, C, SolId, SInfo, Cod, Out>(
        &self,
        _rank: Rank,
    ) -> Result<Acc, CheckpointError>
    where
        Acc: Accumulator<C, SolId, SInfo, Cod, Out>,
        C: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
        SolId: Id,
        SInfo: SolInfo,
        Cod: Codomain<Out>,
        Out: Outcome,
    {
        panic!("NoCheck should not be called to load an experiment.")
    }

    fn get_check_worker<WState: WorkerState>(&self, _proc: &MPIProcess) -> Self::WCheck<WState> {
        panic!("NoCheck should not be called to load an experiment.")
    }
}

#[cfg(feature = "mpi")]
/// An empty [`WorkerState`]
#[derive(Serialize, Deserialize)]
pub struct NoWCheck;

#[cfg(feature = "mpi")]
impl<S: WorkerState> WorkerCheckpointer<S> for NoWCheck {
    type FnStateCheck = NoFuncStateCheck;
    fn init(&mut self, _proc: &MPIProcess) {}
    fn before_load(&mut self, _proc: &MPIProcess) {}
    fn after_load(&mut self, _proc: &MPIProcess) {}
    fn save_state(&self, _state: &S, _rank: Rank) {}
    fn load(&self, _rank: Rank) -> Result<S, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
    fn new_func_state_checkpointer(&self) -> Self::FnStateCheck {
        NoFuncStateCheck
    }
}
