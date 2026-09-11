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
//! ### Function State Checkpointing
//! - **[`FuncStateCheckpointer`]** - For managing the [`FuncState`] of individual function evaluation.
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
//! - All checkpoint data must implement [`Serialize`] and [`Deserialize`].
//! - Checkpointers are configuration-dependent via the [`SaverConfig`] trait.
//! - The [`MessagePack`] implementation is the primary concrete checkpointer for most use cases.
//! - MPI-specific traits are only available with the `mpi` feature.
//! - See [`experiment::Runable`](crate::experiment::Runable) for how checkpointers are integrated into optimization loops.

use crate::{
    Accumulator, FuncState, HasY, Id, Outcome, Runable, SaverConfig, SolInfo, SolutionShape, State, stop::Stop, utils::has_trait::HasState,
};

pub mod messagepack;
pub use messagepack::MessagePack;
use serde::{Deserialize, Serialize};

pub use crate::errors::CheckpointError;


/// Marker trait for checkpointing management of [`FuncState`]s in optimization experiments.
/// This trait provides methods to save, load, and remove the state of individual functions states.
pub trait FuncStateCheckpointer {
    /// Save the [`FuncState`] for a specific function ID in a threaded optimization experiment.
    fn save_func_state<FnState: FuncState, SolId: Id>(&self, id: &SolId, func_state: &FnState);
    /// Loads all the [`FuncState`] for a specific function ID in a threaded optimization experiment.
    fn load_func_state<FnState: FuncState, SolId: Id>(
        &self,
        id: &SolId,
    ) -> Option<(SolId, FnState)>;
    /// Removes the [`FuncState`] for a specific function ID in a threaded optimization experiment, returning whether the state was successfully removed.
    fn remove_func_state<SolId: Id>(&self, id: &SolId) -> Result<bool, CheckpointError>;
    /// Load all the [`FuncState`] returning a vector of function states for each function ID.
    fn load_all_func_state<FnState: FuncState, SolId: Id>(&self) -> Vec<(SolId, FnState)>;
}

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
pub trait Checkpointer<R: Runable>
{
    /// Associated type for [FuncState] checkpointer, used to manage per-evaluation state.
    type FnStateCheck: FuncStateCheckpointer;
    
    /// Configuration type that implements [`SaverConfig`].
    ///
    /// This associated type determines how the checkpointer is configured.
    /// Typically set to [`FolderConfig`](crate::FolderConfig) for file-based checkpointing.
    type Config: SaverConfig;

    /// Creates a new [FuncState] checkpointer for managing per-evaluation state.
    fn new_func_state_checkpointer(&self, ctx: &R::Context) -> Self::FnStateCheck;

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
    fn init(&mut self, runable: &R);

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
    fn before_load(&mut self, ctx: &R::Context);
    /// Performs any necessary cleanup or state preparation after loading a checkpoint.
    /// This method can be used to create a new checkpoint directory after loading an existing one.
    /// It should be called after successfully loading a checkpoint and before resuming optimization.
    fn after_load(&mut self, ctx: &R::Context);
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
    fn save_state<T: HasState>(&self, state: &T, ctx: &R::Context);

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
    fn load<T: HasState>(&self, ctx: &R::Context) -> Result<T, CheckpointError>;
}