use crate::{
    Codomain, Domain, FidOutcome, Fidelity, Id, Objective, Outcome, Stepped,
    checkpointer::{
        DistCheckpointer, WorkerCheckpointer
    },
    experiment::mpi::{
        tools::MPIProcess,
        utils::{DiscardFXMessage, FXMessage, OMessage, XMessage}
    },
    objective::outcome::FuncState,
};

use bincode::serde::Compat;
use mpi::traits::{Communicator, Destination, Source};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

  //--------//
 // WORKER //
//--------//

/// Desribes the different methods of a [`Worker``] process during a [`DistRunable`], a MPI-distributed optimization.
pub trait Worker<SolId,Obj>
where
    Self: Sized,
    SolId: Id,
    Obj:Domain,
{
    type WState: WorkerState;
    fn run(self, proc: &MPIProcess);
}

/// Desribes the internal state of a [`Worker`].
pub trait WorkerState
where
    Self:Sized + Serialize + for<'a> Deserialize<'a>
{}


// BASE WORKER FOR SYNC BATCHES

/// An empty [`WorkerState`]
#[derive(Serialize,Deserialize)]
pub struct NoWState;
impl WorkerState for NoWState{}

/// [`FidWorkerState`] describes the [`WorkerState`] for [`Worker`] computing
/// [`Stepped`] functions. The [`FidWorkerState`] stores the current [`FuncState`]
/// of the function within a [`HashMap`].
#[derive(Serialize,Deserialize)]
#[serde(bound(serialize = "SolId: Serialize",deserialize = "SolId: for<'a> Deserialize<'a>"))]
pub struct FidWState<SolId: Id,FnState: FuncState>(HashMap<SolId, FnState>);
impl<SolId:Id,FnState:FuncState> WorkerState for FidWState<SolId,FnState>{}
impl<SolId: Id,FnState: FuncState> FidWState<SolId,FnState>{
    pub fn new_empty()->Self{FidWState(HashMap::<SolId,FnState>::new())}
}

/// A basic worker for MPI-distributed optimization.
/// A worker is used to evaluate [`Partial`] with the [`Objective`] in parallel.
pub struct BaseWorker<Obj,Cod,Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub objective: Objective<Obj,Cod,Out>,
}
/// Creates a [`BaseWorker`] without any state and checkpointer.
impl <Obj,Cod,Out> BaseWorker<Obj,Cod,Out>
where
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub fn new(objective:Objective<Obj,Cod,Out>)->Self{
        BaseWorker{objective}
    }
}

impl<SolId,Obj,Cod,Out> Worker<SolId,Obj> for BaseWorker<Obj,Cod,Out>
where
    SolId: Id,
    Obj:Domain,
    Cod:Codomain<Out>,
    Out:Outcome,

{
    type WState = NoWState;
    fn run(self, proc: &MPIProcess)
    {
        // Master process is always Rank 0.
        let config = bincode::config::standard();
        loop {
            // Receive X and compute
            let (msg, status): (Vec<u8>, _) = proc.world.process_at_rank(0).receive_vec();
            if status.tag() == 42 {
                break;
            } else {
                let (id_x, _): (Compat<XMessage<SolId, Obj>>, _) =
                    bincode::borrow_decode_from_slice(msg.as_slice(), config).unwrap();
                let msg = id_x.0;
                let id = msg.0;
                let x = msg.1.as_ref();
                let out = self.objective.raw_compute(x);

                // Send results
                let raw_msg: OMessage<SolId, Out> = OMessage(id, out);
                let msg_struct = Compat(raw_msg);
                let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
                proc.world.process_at_rank(0).send(&msg);
            }
        }
        eprintln!(
            "INFO : Process of rank {} exiting worker loop.",
            proc.world.rank()
        );
    }
}

  //----------------//
 //--- FIDELITY ---//
//----------------//

/// A basic fidelity worker for MPI-distributed optimization.
/// A worker is used to evaluate [`Partial`] with the [`Objective`] in parallel.
/// It has a state allowing to 'remember' the previous state of the [`Stepped`] function being evaluated.
pub struct FidWorker<SolId, Obj,Cod,Out,FnState, Check>
where
    SolId: Id,
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    pub objective: Stepped<Obj,Cod,Out,FnState>,
    pub state: FidWState<SolId,FnState>,
    pub check: Option<Check::WCheck<FidWState<SolId,FnState>>>,
}
/// Creates a [`BaseWorker`] without any state and checkpointer.
impl <SolId, Obj,Cod,Out,FnState, Check> FidWorker<SolId, Obj,Cod,Out,FnState,Check>
where
    SolId: Id,
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    pub fn new(objective:Stepped<Obj,Cod,Out,FnState>, check: Option<Check::WCheck<FidWState<SolId,FnState>>>)->Self{
        FidWorker{objective, state:FidWState::new_empty(), check}
    }
}

impl<SolId,Obj,Cod,Out,FnState,Check> Worker<SolId,Obj> for FidWorker<SolId,Obj,Cod,Out,FnState,Check>
where
    SolId: Id,
    Obj:Domain,
    Cod:Codomain<Out>,
    Out:FidOutcome,
    FnState: FuncState,
    Check: DistCheckpointer,
{
    type WState = FidWState<SolId,FnState>;
    fn run(mut self, proc: &MPIProcess)
    {
        // Master process is always Rank 0.
        let config = bincode::config::standard();
        loop {
            // Receive X and compute
            let (msg, status): (Vec<u8>, _) = proc.world.process_at_rank(0).receive_vec();
            // Stop
            if status.tag() == 42 {
                break;
            } 
            // Discard
            else if status.tag() == 104{
                let (id_x, _): (Compat<DiscardFXMessage<SolId>>, _) =
                    bincode::borrow_decode_from_slice(msg.as_slice(), config).unwrap();
                    let msg = id_x.0;
                    let id = msg.0;
                    self.state.0.remove(&id);
            }
            // Checkpoint
            else if status.tag() == 7{
                if let Some(c) = &self.check {c.save_state(&self.state, proc.rank)}
            }
            // Compute
            else {
                let (id_x, _): (Compat<FXMessage<SolId, Obj>>, _) =
                    bincode::borrow_decode_from_slice(msg.as_slice(), config).unwrap();
                let msg = id_x.0;
                let id = msg.0;
                let fid = msg.2;
                match fid{
                    Fidelity::New => {
                        let x = msg.1.as_ref();
                        let (out,state) = self.objective.raw_compute(x,fid,None);
                        self.state.0.insert(id, state);
                        // Send results
                        let raw_msg: OMessage<SolId, Out> = OMessage(id, out);
                        let msg_struct = Compat(raw_msg);
                        let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
                        proc.world.process_at_rank(0).send(&msg);
                    },
                    Fidelity::Resume(_) => {
                        let x = msg.1.as_ref();
                        let state = self.state.0.remove(&id);
                        let (out,state) = self.objective.raw_compute(x,fid,state);
                        if out.get_fidelity().is_partially(){self.state.0.insert(id, state);}
                        // Send results
                        let raw_msg: OMessage<SolId, Out> = OMessage(id, out);
                        let msg_struct = Compat(raw_msg);
                        let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
                        proc.world.process_at_rank(0).send(&msg);
                    },
                    Fidelity::Discard => unreachable!("A Discarded solution should not reach this step."),
                }
            }
        }
        eprintln!("INFO : Process of rank {} exiting worker loop.",proc.world.rank());
    }
}