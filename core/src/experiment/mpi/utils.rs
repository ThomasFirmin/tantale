use crate::{
    Codomain, Computed, Domain, Id, Objective, OptInfo, Outcome, SolInfo, Solution,
    experiment::{mpi::tools::MPIProcess, utils::{BatchResults,PartPair}},
    solution::Batch,
    stop::{ExpStep, Stop}
};

use bincode::{self, config::Configuration, serde::Compat};
use mpi::traits::{Communicator, Destination, Source};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap, sync::{Arc, Mutex}
};

pub type VecArcComputed<SolId, Obj, Cod, Out, SInfo> =
    Vec<Arc<Computed<SolId, Obj, Cod, Out, SInfo>>>;

pub type ArcMutexHash<SolId, Obj, Opt, SInfo> =
    Arc<Mutex<HashMap<SolId, PartPair<SolId, Obj, Opt, SInfo>>>>;

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

pub fn launch_worker<SolId, Obj, Cod, Out>(proc: &MPIProcess, obj_func: &Objective<Obj, Cod, Out>)
where
    SolId: Id,
    Obj: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
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
            let out = obj_func.raw_compute(x);

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

/// A structure containing utilitaries to send [`Partial`] to workers.
pub struct SendRecParam<'a, Obj, Opt, SInfo, SolId>
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    pub config: Configuration,
    pub proc: &'a MPIProcess,
    pub idle: &'a mut Vec<i32>,
    pub waiting: &'a mut HashMap<SolId, PartPair<SolId, Obj, Opt, SInfo>>,
}

/// A structure containing utilitaries to send [`Partial`] to workers while using multi-threading.
pub struct ThrSendRecParam<'a, Obj, Opt, SInfo, SolId>
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    pub config: Configuration,
    pub proc: &'a MPIProcess,
    pub idle: Arc<Mutex<Vec<i32>>>,
    pub waiting: ArcMutexHash<SolId, Obj, Opt, SInfo>,
}

/// Send an Obj [`Solution`] to a worker
pub fn send_to_worker<'a, SolId, Obj, Opt, SInfo>(
    params: &mut SendRecParam<'a, Obj, Opt, SInfo, SolId>,
    pair: PartPair<SolId, Obj, Opt, SInfo>,
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let has_idl = !params.idle.is_empty();
    if has_idl {
        let sid = pair.0.id;
        let rank = params.idle.pop().unwrap();
        let raw_msg: XMessage<SolId, Obj> = XMessage(sid, pair.0.get_x());
        let msg_struct = Compat(raw_msg);
        let msg = bincode::encode_to_vec(msg_struct, params.config).unwrap();
        params.waiting.insert(sid, pair);
        params
            .proc
            .world
            .process_at_rank(rank)
            .send_with_tag(&msg, 1);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
pub fn fill_workers<'a, SolId, Obj, Opt, SInfo, Info, St>(
    params: &mut SendRecParam<'a, Obj, Opt, SInfo, SolId>,
    stop: Arc<Mutex<St>>,
    batch:&Batch<SolId,Obj,Opt,SInfo,Info>,
    idx: usize,
) -> usize
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    St: Stop,
    
{
    let mut i: usize = idx;
    let mut st = stop.lock().unwrap();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < batch.size() && !st.stop() {
        let has_idle = send_to_worker(params,batch.index(i));
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
    params: &mut ThrSendRecParam<Obj, Opt, SInfo, SolId>,
    pair: PartPair<SolId,Obj,Opt,SInfo>
) -> bool
where
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    let mut idl = params.idle.lock().unwrap();
    let has_idl = !idl.is_empty();
    if has_idl {
        let sid = pair.0.id;
        let rank = idl.pop().unwrap();
        let x = pair.0.get_x();
        let msg_struct = Compat((sid, x.as_ref()));
        let msg = bincode::encode_to_vec(msg_struct, params.config).unwrap();
        params.waiting.lock().unwrap().insert(sid, pair);
        params
            .proc
            .world
            .process_at_rank(rank)
            .send_with_tag(&msg, 1);
    }
    has_idl
}

// Send as much solutions as possible to idle workers without waiting for a result.
pub fn par_fill_workers<SolId, Obj, Opt, SInfo, Info, St>(
    params: &mut ThrSendRecParam<Obj, Opt, SInfo, SolId>,
    stop: Arc<Mutex<St>>,
    batch:Batch<SolId,Obj,Opt,SInfo,Info>,
    idx: usize,
) -> usize
where
    St: Stop,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info:OptInfo,
    SolId: Id,
{
    let mut i = idx;
    let mut st = stop.lock().unwrap();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < batch.size() && !st.stop() {
        let has_idle = par_send_to_worker(params, batch.index(i));
        if has_idle {
            st.update(ExpStep::Distribution);
            i += 1;
        } else {
            at_least_one_idle = false
        }
    }
    i
}

pub fn receive_obj_computed<'a, Obj, Opt, SInfo, Info, SolId, Cod, Out>(
    params: &mut SendRecParam<'a, Obj, Opt, SInfo, SolId>,
    res: &mut BatchResults<SolId, Obj, Opt, Cod, Out, SInfo,Info>,
    ob: &Objective<Obj, Cod, Out>,
) where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    // Recv / sendv loop
    let (bytes, status): (Vec<u8>, _) = params.proc.world.any_process().receive_vec();
    params.idle.push(status.source_rank());
    let (bytes, _): (Compat<OMessage<SolId, Out>>, _) =
        bincode::decode_from_slice(bytes.as_slice(), params.config).unwrap();
    let msg = bytes.0;
    let id = msg.0;
    let out = msg.1;
    let y = Arc::new(ob.codomain.get_elem(&out));
    let out = Arc::new(out);
    let pair = params.waiting.remove(&id).unwrap();
    res.add(pair, out, y);
}
