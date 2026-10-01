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

use std::path::PathBuf;

use crate::{FuncState, Problem, SolId, StepId, utils::has_trait::HasState};

pub mod messagepack;
pub use messagepack::MessagePack;

pub use crate::errors::CheckpointerError;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CheckpointerTargets {
    Checkpoint,
    Backup(PathBuf),
    Optimizer,
    Accumulator,
    Stop,
    Evaluator,
    Config,
    FuncFolder,
    WorkerFolder,
    Other(PathBuf),
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
pub trait Checkpointer<P: Problem, Context> {
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
    fn init(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;

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
    fn before_load(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;
    /// Performs any necessary cleanup or state preparation after loading a checkpoint.
    /// This method can be used to create a new checkpoint directory after loading an existing one.
    /// It should be called after successfully loading a checkpoint and before resuming optimization.
    fn after_load(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;
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
    fn save_state<T: HasState>(
        &self,
        state: &T,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

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
    /// Returns [`CheckpointerError`] if:
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
    fn load<T: HasState>(
        &self,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<T, CheckpointerError>;

    /// Loads all the states of type `T` from a checkpoint folder.
    ///
    /// This method is useful for loading multiple saved states, such as all function states.
    fn load_folder<T: HasState>(
        &self,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<Vec<T>, CheckpointerError>;


    fn save_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: &Id,
        state: &FnState,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    fn remove_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: &Id,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    fn remove_all_previous_fn_states<Id: StepId, FnState: FuncState>(
        &self,
        id: &Id,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>{
        let previous = id.previous_id();
        let res = self.remove_fn_state::<Id, FnState>(&previous, ctx);
        if res.is_ok() {
            self.remove_all_previous_fn_states::<Id, FnState>(&previous, ctx)
        } else {
            res
        }
    }

    fn load_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: Id,
        ctx: &Context,
    ) -> Result<(Id, FnState), CheckpointerError>;

    fn load_fn_state_folder<Id: SolId, FnState: FuncState>(
        &self,
        ctx: &Context,
    ) -> Result<Vec<(Id, FnState)>, CheckpointerError>;

    /// Removes the checkpointed state at a specific target.
    fn remove(
        &self,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    /// Creates a backup of the checkpointed state at a specific target.
    fn backup(
        &self,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<PathBuf, CheckpointerError>;
}