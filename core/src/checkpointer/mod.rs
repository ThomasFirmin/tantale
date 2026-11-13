//! Checkpoint used by experiment runners.
//!
//! The [`Checkpointer`] trait saves and loads states selected by
//! [`CheckpointerTargets`]. Its methods return [`CheckpointerError`] on storage
//! or serialization failures. [`MessagePack`] is the concrete implementation
//! provided by this module.
//!
//! Checkpointers are passed to [`Experiment`](crate::experiment::Experiment)
//! or [`ParExperiment`](crate::experiment::ParExperiment), which save the
//! state required by the runner. Function state for multi-step objectives can
//! also be saved and loaded by solution identifier.
//!
//! See the crate-level documentation for setup and execution examples.

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
    Experiment,
    Config,
    FuncFolder,
    WorkerFolder,
    Other(PathBuf),
}

/// Interface for saving and loading experiment state.
///
/// The runner selects a [`CheckpointerTargets`] value for each state it
/// persists. Implementations return [`CheckpointerError`] when an operation
/// fails.
pub trait Checkpointer<P: Problem, Context> {
    /// Prepare storage for a new experiment.
    fn init(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;

    /// Prepare storage for loading a previous experiment.
    fn before_load(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;

    /// Finalize storage setup after restoring an experiment.
    fn after_load(&mut self, p: &P, context: &Context) -> Result<(), CheckpointerError>;

    /// Save a state value under the requested target.
    fn save_state<T: HasState>(
        &self,
        state: &T,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    /// Load a state value previously stored under the requested target.
    ///
    /// Returns [`CheckpointerError`] if the target cannot be read or decoded.
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

    /// Saves a specific [`FuncState`] associated with a given [`SolId`] to the checkpoint folder.
    fn save_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: &Id,
        state: &FnState,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    /// Removes a specific [`FuncState`] associated with a given [`SolId`] from the checkpoint folder.
    fn remove_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: &Id,
        ctx: &Context,
    ) -> Result<(), CheckpointerError>;

    /// Recursively removes all previous function states associated with a given [`SolId`].
    fn remove_all_previous_fn_states<Id: StepId, FnState: FuncState>(
        &self,
        id: &Id,
        ctx: &Context,
    ) -> Result<(), CheckpointerError> {
        let res = self.remove_fn_state::<Id, FnState>(id, ctx);
        if res.is_ok() {
            self.remove_all_previous_fn_states::<Id, FnState>(&id.previous_id(), ctx)
        } else {
            res
        }
    }

    /// Loads a [`FuncState`] associated with a specific [`SolId`] from a checkpoint folder.
    fn load_fn_state<Id: SolId, FnState: FuncState>(
        &self,
        id: Id,
        ctx: &Context,
    ) -> Result<(Id, FnState), CheckpointerError>;

    /// Loads all the states of type [`FuncState`] from a checkpoint folder.
    fn load_fn_state_folder<Id: SolId, FnState: FuncState>(
        &self,
        ctx: &Context,
    ) -> Result<Vec<(Id, FnState)>, CheckpointerError>;

    /// Lists all the [`SolId`]s that have associated [`FuncState`]s stored in the checkpoint folder.
    fn list_ids<Id: SolId>(&self, ctx: &Context) -> Result<Vec<Id>, CheckpointerError>;

    /// Removes the checkpointed state at a specific target.
    fn remove(&self, target: &CheckpointerTargets, ctx: &Context) -> Result<(), CheckpointerError>;

    /// Creates a backup of the checkpointed state at a specific target.
    fn backup(
        &self,
        target: &CheckpointerTargets,
        ctx: &Context,
    ) -> Result<PathBuf, CheckpointerError>;

    /// Add a suffix to a [`CheckpointerTargets`] to create a new target for saving state.
    fn append_to_target(
        &self,
        target: &CheckpointerTargets,
        suffix: &str,
    ) -> Result<PathBuf, CheckpointerError>;

    /// Checks if the checkpointer is configured for parallel file system (PFS) usage, which may affect how checkpoints are saved and loaded.
    fn is_pfs(&self) -> bool;

    /// Sets the checkpointer's configuration for parallel file system (PFS) usage.
    fn set_pfs(&mut self, pfs: bool);
}
