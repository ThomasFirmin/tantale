use crate::{
    FidOutcome, Fidelity, Id, Objective, Outcome, Stepped,
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::{
        basics::FuncStatePool,
        mpi::utils::{DiscardFXMessage, FXMessage, MPIProcess, Msg, OMessage, XMessage, send_msg},
    },
    objective::{Step, outcome::FuncState},
};

use core::panic;
use mpi::traits::{Communicator, Source};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

//--------//
// WORKER //
//--------//

/// Describes a [`Worker`] used in an MPI-distributed experiment ([`MPIExperiment)`](crate::MPIExperiment)).
/// A [`Worker`] is mostly used to evaluate [`Uncomputed`](crate::solution::Uncomputed)
/// [`Raw`](crate::Solution::Raw) solutions.
///
/// It has an associated [`WorkerState`] type, allowing to store internal state information
/// during the evaluation process. Allowing per-[`Worker`] checkpointing
/// via [`WorkerCheckpointer`](crate::checkpointer::WorkerCheckpointer).
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

/// [`NoWState`] describes a [`WorkerState`] without any internal state.
#[derive(Serialize, Deserialize)]
pub struct NoWState;
impl WorkerState for NoWState {}

/// [`FidWState`] describes the [`WorkerState`] for [`Worker`] computing
/// [`Stepped`] functions. The [`FidWState`] stores the current [`FuncState`]
/// of the function within a [`HashMap`]. The keys of the map are the
/// solution [`Id`]s, and the values are the corresponding [`FuncState`]s.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "SolId: Serialize",
    deserialize = "SolId: for<'a> Deserialize<'a>"
))]
pub struct FidWState<SolId: Id, FnState: FuncState, FuncPool: FuncStatePool<FnState, SolId>> {
    #[serde(skip)]
    pub pool: FuncPool,
    _id: PhantomData<SolId>,
    _fnstate: PhantomData<FnState>,
}
impl<SolId: Id, FnState: FuncState, FuncPool: FuncStatePool<FnState, SolId>> WorkerState
    for FidWState<SolId, FnState, FuncPool>
{
}
impl<SolId: Id, FnState: FuncState, FuncPool: FuncStatePool<FnState, SolId>>
    FidWState<SolId, FnState, FuncPool>
{
    pub fn new_empty() -> Self {
        FidWState {
            pool: FuncPool::default(),
            _id: PhantomData,
            _fnstate: PhantomData,
        }
    }
    pub fn new(pool: FuncPool) -> Self {
        FidWState {
            pool,
            _id: PhantomData,
            _fnstate: PhantomData,
        }
    }
}

/// A basic [`Worker`] for MPI-distributed optimization.
/// A [`BaseWorker`] is used to evaluate [`Raw`](crate::Solution::Raw) with a given [`Objective`].
pub struct BaseWorker<'a, Raw, Out>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    Out: Outcome,
{
    pub proc: &'a MPIProcess,
    pub objective: Objective<Raw, Out>,
}

impl<'a, Raw, Out> BaseWorker<'a, Raw, Out>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    Out: Outcome,
{
    /// Creates a [`BaseWorker`] without any state and checkpointer.
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
    /// No internal state for the base worker.
    type WState = NoWState;
    /// Runs the main loop of the [`BaseWorker`].
    /// It waits for messages from the master process (rank 0),
    /// computes the corresponding outputs with the given [`Objective`],
    /// and sends back the results to the master process.
    ///
    /// If the received message has a tag of `42`, the worker exits the loop and terminates.
    /// Otherwise, it processes the received [`Raw`](crate::Solution::Raw) as an [`XMessage`], computes the output,
    /// and sends back an [`OMessage`] containing the results.
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
/// A [`FidWorker`] is used to evaluate [`Raw`](crate::Solution::Raw) with a given [`Stepped`] [`Objective`].
/// It maintains an internal state mapping solution [`Id`]s to their corresponding [`FuncState`](crate::objective::outcome::FuncState`).
/// This allows the worker to handle computations that may require multiple [`Step`]s.
/// The worker can also optionally utilize a [`WorkerCheckpointer`] to save its state during the computation process.
pub struct FidWorker<'a, SolId, Raw, Out, FnState, Check, FnPool>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
    FnPool: FuncStatePool<FnState, SolId>,
{
    pub proc: &'a MPIProcess,
    pub objective: Stepped<Raw, Out, FnState>,
    pub state: FidWState<SolId, FnState, FnPool>,
    pub check: Option<Check::WCheck<FidWState<SolId, FnState, FnPool>>>,
}

impl<'a, SolId, Raw, Out, FnState, Check, FnPool>
    FidWorker<'a, SolId, Raw, Out, FnState, Check, FnPool>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
    FnPool: FuncStatePool<FnState, SolId>,
{
    /// Creates a new [`FidWorker`] with the given parameters.
    pub fn new(
        proc: &'a MPIProcess,
        objective: Stepped<Raw, Out, FnState>,
        pool: FnPool,
        check: Option<Check::WCheck<FidWState<SolId, FnState, FnPool>>>,
    ) -> Self {
        let state = FidWState::new(pool);
        FidWorker {
            proc,
            objective,
            state,
            check,
        }
    }
}

impl<'a, SolId, Raw, Out, FnState, Check, FnPool> Worker<SolId>
    for FidWorker<'a, SolId, Raw, Out, FnState, Check, FnPool>
where
    Raw: Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
    FnPool: FuncStatePool<FnState, SolId>,
{
    /// The internal state type for the fidelity worker.
    type WState = FidWState<SolId, FnState, FnPool>;
    /// Runs the main loop of the fidelity worker.
    /// It waits for messages from the master process (rank 0),
    /// computes the corresponding outputs with the given [`Stepped`] [`Objective`],
    /// and sends back the results to the master process.
    /// It is able to recover [`FuncState`] from previous [`Step`]s of [`Step::Partially`] [`Uncomputed`](crate::solution::Uncomputed) solutions
    /// evaluated within this [`Worker`].
    ///
    /// If the received message has a tag of `42`, the worker exits the loop and terminates.
    /// If the tag is `104`, it discards the internal state associated with the given solution [`Id`].
    /// If the tag is `7`, it checkpoints its internal state using the provided [`WorkerCheckpointer`].
    /// Otherwise, it processes the received [`Raw`](crate::Solution::Raw) as an [`FXMessage`], computes the output,
    /// and sends back an [`OMessage`] containing the results.
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
                            self.state.pool.insert(id, state);
                        } else {
                            self.state.pool.remove(&id);
                        }
                        // Send results
                        send_msg(self.proc, OMessage(id, out), 0, 0, config);
                    }
                    Step::Partially(_) => {
                        let state = self.state.pool.retrieve(&id);
                        let (out, state) = self.objective.compute(x, fid, state);
                        if out.get_step().0 > 0 {
                            // if > 0 => Partially
                            self.state.pool.insert(id, state);
                        } else {
                            self.state.pool.remove(&id);
                        }
                        // Send results
                        send_msg(self.proc, OMessage(id, out), 0, 0, config);
                    }
                    _ => {
                        unreachable!(
                            "A Discarded, Evaluated or Errored solution should not reach this step."
                        )
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
                self.state.pool.remove(&msg.0);
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
