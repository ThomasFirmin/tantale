use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::mpi::utils::{
        send_msg, DiscardFXMessage, FXMessage, MPIProcess, Msg, OMessage, XMessage,
    },
    objective::{outcome::FuncState, Step},
    FidOutcome, Fidelity, Id, Objective, Outcome, Stepped,
};

use core::panic;
use mpi::traits::{Communicator, Source};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

//--------//
// WORKER //
//--------//

/// Desribes the different methods of a [`Worker``] process during a [`DistRunable`], a MPI-distributed optimization.
pub trait Worker<SolId: Id> {
    type WState: WorkerState;
    fn run(self);
}

/// Desribes the internal state of a [`Worker`].
pub trait WorkerState
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
{
}

// BASE WORKER FOR SYNC BATCHES

/// An empty [`WorkerState`]
#[derive(Serialize, Deserialize)]
pub struct NoWState;
impl WorkerState for NoWState {}

/// [`FidWorkerState`] describes the [`WorkerState`] for [`Worker`] computing
/// [`Stepped`] functions. The [`FidWorkerState`] stores the current [`FuncState`]
/// of the function within a [`HashMap`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "SolId: Serialize",
    deserialize = "SolId: for<'a> Deserialize<'a>"
))]
pub struct FidWState<SolId: Id, FnState: FuncState>(HashMap<SolId, FnState>);
impl<SolId: Id, FnState: FuncState> WorkerState for FidWState<SolId, FnState> {}
impl<SolId: Id, FnState: FuncState> FidWState<SolId, FnState> {
    pub fn new_empty() -> Self {
        FidWState(HashMap::<SolId, FnState>::new())
    }
}

/// A basic worker for MPI-distributed optimization.
/// A worker is used to evaluate [`Partial`] with the [`Objective`] in parallel.
pub struct BaseWorker<'a, Raw, Out>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    Out: Outcome,
{
    pub proc: &'a MPIProcess,
    pub objective: Objective<Raw, Out>,
}
/// Creates a [`BaseWorker`] without any state and checkpointer.
impl<'a, Raw, Out> BaseWorker<'a, Raw, Out>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    Out: Outcome,
{
    pub fn new(objective: Objective<Raw, Out>, proc: &'a MPIProcess) -> Self {
        BaseWorker { proc, objective }
    }
}

impl<'a, SolId, Raw, Out> Worker<SolId> for BaseWorker<'a, Raw, Out>
where
    SolId: Id,
    Raw: Serialize + for<'de> Deserialize<'de>,
    Out: Outcome,
{
    type WState = NoWState;
    fn run(self) {
        // Master process is always Rank 0.
        let config = bincode::config::standard();
        loop {
            // Receive X and compute
            let (raw, status): (Vec<u8>, _) = self.proc.world.process_at_rank(0).receive_vec();
            if status.tag() == 42 {
                break;
            } else {
                let msg = XMessage::<SolId, Raw>::from_bytes(raw, config);
                let id = msg.0;
                let x = msg.1;
                let out = self.objective.compute(x);

                // Send results
                send_msg(self.proc, OMessage(id, out), 0, 0, config);
            }
        }
        eprintln!(
            "INFO : Process of rank {} exiting worker loop.",
            self.proc.world.rank()
        );
    }
}

//----------------//
//--- FIDELITY ---//
//----------------//

/// A basic fidelity worker for MPI-distributed optimization.
/// A worker is used to evaluate [`Partial`] with the [`Objective`] in parallel.
/// It has a state allowing to 'remember' the previous state of the [`Stepped`] function being evaluated.
pub struct FidWorker<'a, SolId, Raw, Out, FnState, Check>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    pub proc: &'a MPIProcess,
    pub objective: Stepped<Raw, Out, FnState>,
    pub state: FidWState<SolId, FnState>,
    pub check: Option<Check::WCheck<FidWState<SolId, FnState>>>,
}
/// Creates a [`BaseWorker`] without any state and checkpointer.
impl<'a, SolId, Raw, Out, FnState, Check> FidWorker<'a, SolId, Raw, Out, FnState, Check>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    pub fn new(
        objective: Stepped<Raw, Out, FnState>,
        check: Option<Check::WCheck<FidWState<SolId, FnState>>>,
        proc: &'a MPIProcess,
    ) -> Self {
        FidWorker {
            proc,
            objective,
            state: FidWState::new_empty(),
            check,
        }
    }
}

impl<'a, SolId, Raw, Out, FnState, Check> Worker<SolId>
    for FidWorker<'a, SolId, Raw, Out, FnState, Check>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    type WState = FidWState<SolId, FnState>;
    fn run(mut self) {
        // Master process is always Rank 0.
        let config = bincode::config::standard();
        loop {
            // Receive X and compute
            let (raw, status): (Vec<u8>, _) = self.proc.world.process_at_rank(0).receive_vec();
            let tag = status.tag();
            // Receive point to compute
            if tag == 0 {
                // 0: id, 1: X, 2: Fidelity
                let msg = FXMessage::<SolId, Raw>::from_bytes(raw, config);
                let x = msg.1;
                let id = msg.0;
                let step: Step = msg.2.into();
                let fid: Fidelity = msg.3;
                match step {
                    Step::Pending => {
                        let (out, state) = self.objective.compute(x, fid, None);
                        if out.get_step().0 > 0 {
                            self.state.0.insert(id, state);
                        }
                        // Send results
                        send_msg(self.proc, OMessage(id, out), 0, 0, config);
                    }
                    Step::Partially(_) => {
                        let state = self.state.0.remove(&id);
                        let (out, state) = self.objective.compute(x, fid, state);
                        if out.get_step().0 > 0 {
                            // if > 0, Partially
                            self.state.0.insert(id, state);
                        }
                        // Send results
                        send_msg(self.proc, OMessage(id, out), 0, 0, config);
                    }
                    _ => {
                        unreachable!("A Discarded, Evaluated or Errored solution should not reach this step.")
                    }
                }
            }
            // Stop
            else if tag == 42 {
                break;
            }
            // Discard
            else if tag == 104 {
                let msg = DiscardFXMessage::from_bytes(raw, config);
                self.state.0.remove(&msg.0);
            }
            // Checkpoint
            else if tag == 7 {
                if let Some(c) = &self.check {
                    c.save_state(&self.state, self.proc.rank)
                }
            }
            // Compute
            else {
                panic!(
                    "Unknown tag ({}) for message send to worker {}",
                    tag, self.proc.rank
                );
            }
        }
        eprintln!(
            "INFO : Process of rank {} exiting worker loop.",
            self.proc.world.rank(),
        );
    }
}
