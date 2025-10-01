use crate::{
    ArcVecArc, Codomain, Computed, Domain, Id, MPI_WORLD, Objective, Outcome, Partial, SolInfo, Solution, stop::{ExpStep, Stop}
};

use bincode::{self, config::Configuration, serde::Compat};
use mpi::{
    topology::SimpleCommunicator,
    traits::{Communicator, Destination, Source},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub type SolPair<SolId, Obj, Opt, SInfo> = (
    Arc<Partial<SolId, Obj, SInfo>>,
    Arc<Partial<SolId, Opt, SInfo>>,
);

pub type VecArcComputed<SolId, Obj, Cod, Out, SInfo> = Vec<Arc<Computed<SolId, Obj, Cod, Out, SInfo>>>;

pub type ArcMutexHash<SolId, Obj, Opt, SInfo> = Arc<Mutex<HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>>>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, SolId:Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct XMessage<SolId: Id, Dom: Domain>(pub SolId, pub Arc<[Dom::TypeDom]>);

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct OMessage<SolId: Id, Out: Outcome>(pub SolId, pub Out);

//_______ //
// WORKER //
//_______ //

pub fn launch_worker<SolId, Obj, Cod, Out>(obj_func: &Objective<Obj, Cod, Out>)
where
    SolId: Id,
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    // Master process is always Rank 0.
    let world = MPI_WORLD.get().unwrap();
    let config = bincode::config::standard();
    loop {
        // Receive X and compute
        let (msg, status): (Vec<u8>, _) = world.process_at_rank(0).receive_vec();
        if status.tag() == -1 {
            break;
        }
        let (id_x, _): (Compat<XMessage<SolId, Obj>>, _) =
            bincode::borrow_decode_from_slice(msg.as_slice(), config).unwrap();
        let msg = id_x.0;
        let id = msg.0;
        let x = msg.1.as_ref();
        let out = obj_func.raw_compute(x);

        // Send results
        let raw_msg: OMessage<SolId, Out> = OMessage(id, out);
        let msg_struct = Compat(raw_msg);
        let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
        world.process_at_rank(0).send(&msg);
    }
    std::process::exit(0);
}

// Send an Obj solution to a worker
pub fn send_to_worker<SolId, Obj, Opt, SInfo>(
    world: &SimpleCommunicator,
    idle: &mut Vec<i32>,
    config: Configuration,
    sobj: Arc<Partial<SolId, Obj, SInfo>>,
    sopt: Arc<Partial<SolId, Opt, SInfo>>,
    waiting: &mut HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>>,
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let has_idl = !idle.is_empty();
    if has_idl {
        let rank = idle.pop().unwrap();
        let raw_msg: XMessage<SolId, Obj> = XMessage(sobj.id, sobj.get_x());
        let msg_struct = Compat(raw_msg);
        let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
        waiting.insert(sobj.id, (sobj, sopt));
        world.process_at_rank(rank).send_with_tag(&msg, 1);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
pub fn fill_workers<SolId, Obj, Opt, SInfo, St>(
    idle: &mut Vec<i32>,
    stop: Arc<Mutex<St>>,
    in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    idx: usize,
    waiting: &mut HashMap<SolId, SolPair<SolId, Obj, Opt, SInfo>>,
    config: Configuration,
) -> usize
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut i = idx;
    let mut st = stop.lock().unwrap();
    let world = MPI_WORLD.get().unwrap();
    let length = in_obj.len();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < length && !st.stop() {
        let has_idle = send_to_worker(
            world,
            idle,
            config,
            in_obj[i].clone(),
            in_opt[i].clone(),
            waiting,
        );
        if has_idle {
            st.update(ExpStep::Distribution);
            i += 1;
        } else {
            at_least_one_idle = false
        }
    }
    i
}

pub fn par_send_to_worker<SolId, Obj, Opt, SInfo>(
    world: &SimpleCommunicator,
    idle: Arc<Mutex<Vec<i32>>>,
    config: Configuration,
    sobj: Arc<Partial<SolId, Obj, SInfo>>,
    sopt: Arc<Partial<SolId, Opt, SInfo>>,
    waiting: ArcMutexHash<SolId, Obj, Opt, SInfo>,
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut idl = idle.lock().unwrap();
    let has_idl = !idl.is_empty();
    if has_idl {
        let rank = idl.pop().unwrap();
        let id = sobj.id;
        let x = sobj.get_x();
        let msg_struct = Compat((id, x.as_ref()));
        let msg = bincode::encode_to_vec(msg_struct, config).unwrap();
        waiting.lock().unwrap().insert(id, (sobj, sopt));
        world.process_at_rank(rank).send_with_tag(&msg, 1);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
pub fn par_fill_workers<SolId, Obj, Opt, SInfo, St>(
    idle: Arc<Mutex<Vec<i32>>>,
    stop: Arc<Mutex<St>>,
    in_obj: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    in_opt: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    idx: usize,
    waiting: ArcMutexHash<SolId, Obj, Opt, SInfo>,
    config: Configuration,
) -> usize
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut i = idx;
    let mut st = stop.lock().unwrap();
    let world = MPI_WORLD.get().unwrap();
    let length = in_obj.len();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < length && !st.stop() {
        let has_idle = par_send_to_worker(
            world,
            idle.clone(),
            config,
            in_obj[i].clone(),
            in_opt[i].clone(),
            waiting.clone(),
        );
        if has_idle {
            st.update(ExpStep::Distribution);
            i += 1;
        } else {
            at_least_one_idle = false
        }
    }
    i
}