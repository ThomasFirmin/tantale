use crate::{GlobalParameters, SaverConfig, experiment::Evaluate, optimizer::OptState, stop::Stop};
#[cfg(feature = "mpi")]
use crate::{
    config::NoConfig,
    experiment::mpi::{utils::MPIProcess, worker::WorkerState},
};

#[cfg(feature = "mpi")]
use mpi::Rank;

pub mod messagepack;
pub use messagepack::MessagePack;
#[cfg(feature = "mpi")]
use serde::{Deserialize, Serialize};

pub use crate::errors::CheckpointError;

/// A [`Checkpointer`] is used to frequently create a checkpoint from the states of the
/// the various elements ([`Stop`], [`OptState`], [`Evaluate`], [`GlobalParameters`]), used during the optimization process.
pub trait Checkpointer
where
    Self: Sized,
{
    type Config: SaverConfig;
    /// Initializes a non-existing checkpoint.
    fn init(&mut self);
    fn after_load(&mut self);
    /// Save the current state of the optimization.
    fn save_state<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
    );
    /// Load an existing checkpoint.
    fn load<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Eval), CheckpointError>;
    /// Load the [`Stop`] from an existing checkpoint.
    fn load_stop<St: Stop>(&self) -> Result<St, CheckpointError>;
    /// Load the [`OptState`] from an existing checkpoint.
    fn load_optimizer<OState: OptState>(&self) -> Result<OState, CheckpointError>;
    /// Load the [`Evaluate`] from an existing checkpoint.
    fn load_evaluate<Eval: Evaluate>(&self) -> Result<Eval, CheckpointError>;
    /// Load the [`GlobalParameters`] from an existing checkpoint.
    fn load_parameters(&self) -> Result<GlobalParameters, CheckpointError>;
}

/// A [`ThrCheckpointer`] is a [`Checkpointer`] for threaded experiment where each threat has it's own state.
pub trait ThrCheckpointer: Checkpointer
where
    Self: Sized,
{
    type Config: SaverConfig;
    /// Initializes a non-existing checkpoint where `k` is the number of threads with their own [`Evaluate`] state.
    fn init_thr(&mut self);
    fn after_load_thr(&mut self);
    /// Save the current state of the optimization.
    fn save_state_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        thread: usize,
    );
    /// Load an existing checkpoint.
    fn load_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Vec<Eval>), CheckpointError>;
    /// Load the [`Stop`] from an existing checkpoint.
    fn load_stop_thr<St: Stop>(&self) -> Result<St, CheckpointError>;
    /// Load the [`OptState`] from an existing checkpoint.
    fn load_optimizer_thr<OState: OptState>(&self) -> Result<OState, CheckpointError>;
    /// Load the [`Evaluate`] from an existing checkpoint for the k-th thread.
    fn load_evaluate_thr<Eval: Evaluate>(&self) -> Result<Vec<Eval>, CheckpointError>;
    /// Load the [`GlobalParameters`] from an existing checkpoint.
    fn load_parameters_thr(&self) -> Result<GlobalParameters, CheckpointError>;
}

#[cfg(feature = "mpi")]
/// A [`WorkerCheckpointer`] is used to create a checkpoint from the [`WorkerState`] during the optimization process.
pub trait WorkerCheckpointer<WState>
where
    Self: Sized,
    WState: WorkerState,
{
    /// Initializes a non-existing checkpoint.
    fn init(&mut self, proc: &MPIProcess);
    /// This method should be executed after a [`load`](WorkerCheckpointer::load).
    /// In that case it replaces the [`init`](WorkerCheckpointer::init), when the checkpoint already exists. While
    /// [`init`](WorkerCheckpointer::init) initializes an non-existing checkpoint.
    fn after_load(&mut self, proc: &MPIProcess);
    /// Save the current state of a [`WorkerState`] from a [`Worker`].
    fn save_state(&self, state: &WState, rank: Rank);
    /// Load the [`WorkerState`] from an existing checkpoint.
    fn load(&self, rank: Rank) -> Result<WState, CheckpointError>;
}

#[cfg(feature = "mpi")]
/// Distributed version of [Checkpointer].
pub trait DistCheckpointer: Checkpointer
where
    Self: Sized,
{
    type WCheck<WState: WorkerState>: WorkerCheckpointer<WState>;
    /// Initializes a non-existing checkpoint.
    fn init_dist(&mut self, proc: &MPIProcess);
    fn after_load_dist(&mut self, proc: &MPIProcess);
    /// Define an initialization for [`Workers`] that do not have a [`Checkpointer`].
    fn no_check_init(proc: &MPIProcess);
    /// Save the current state of the optimization.
    fn save_state_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        rank: Rank,
    );
    /// Load an existing checkpoint.
    fn load_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        rank: Rank,
    ) -> Result<(OState, St, Eval), CheckpointError>;
    /// Load the [`Stop`] from an existing checkpoint.
    fn load_stop_dist<St: Stop>(&self, rank: Rank) -> Result<St, CheckpointError>;
    /// Load the [`OptState`] from an existing checkpoint.
    fn load_optimizer_dist<OState: OptState>(&self, rank: Rank) -> Result<OState, CheckpointError>;
    /// Load the [`Evaluate`] from an existing checkpoint.
    fn load_evaluate_dist<Eval: Evaluate>(&self, rank: Rank) -> Result<Eval, CheckpointError>;
    /// Load the [`GlobalParameters`] from an existing checkpoint.
    fn load_parameters_dist(&self, rank: Rank) -> Result<GlobalParameters, CheckpointError>;
    /// Return the checkpointer for the [`Worker`].
    fn get_check_worker<WState: WorkerState>(&self, proc: &MPIProcess) -> Self::WCheck<WState>;
}

#[cfg(feature = "mpi")]
/// An empty [`WorkerState`]
#[derive(Serialize, Deserialize)]
pub struct NoCheck;

#[cfg(feature = "mpi")]
impl Checkpointer for NoCheck {
    type Config = NoConfig;

    fn init(&mut self) {}

    fn after_load(&mut self) {}

    fn save_state<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        _state: &OState,
        _stop: &St,
        _eval: &Eval,
    ) {
    }

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
}

#[cfg(feature = "mpi")]
impl DistCheckpointer for NoCheck {
    type WCheck<WState: WorkerState> = NoWCheck;

    fn init_dist(&mut self, _proc: &MPIProcess) {}
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
    fn init(&mut self, _proc: &MPIProcess) {}
    fn after_load(&mut self, _proc: &MPIProcess) {}
    fn save_state(&self, _state: &S, _rank: Rank) {}
    fn load(&self, _rank: Rank) -> Result<S, CheckpointError> {
        panic!("NoCheck should not be called to load an experiment.")
    }
}
