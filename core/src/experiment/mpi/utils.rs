use crate::{
    experiment::{
        mpi::tools::MPIProcess,
        utils::{BatchResults, PartPair},
    },
    solution::Batch,
    stop::{ExpStep, Stop},
    Codomain, Computed, Domain, Id, Objective, OptInfo, Outcome, Partial, SolInfo,
};

use bincode::{self, config::Configuration, serde::Compat};
use mpi::traits::{Communicator, Destination, Source};
use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub type VecArcComputed<PSol, SolId, Dom, Cod, Out, SInfo> =
    Vec<Arc<Computed<PSol, SolId, Dom, Cod, Out, SInfo>>>;

pub type ArcMutexHash<PSolA, PSolB, SolId> = Arc<Mutex<HashMap<SolId, PartPair<PSolA, PSolB>>>>;

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
pub struct SendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    pub config: Configuration,
    pub proc: &'a MPIProcess,
    pub idle: &'a mut Vec<i32>,
    pub waiting: &'a mut HashMap<SolId, PartPair<PSol, PSol::Twin<Opt>>>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
}

impl<'a, PSol, Obj, Opt, SolId, SInfo> SendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    pub fn new(
        config: Configuration,
        proc: &'a MPIProcess,
        idle: &'a mut Vec<i32>,
        waiting: &'a mut HashMap<SolId, PartPair<PSol, PSol::Twin<Opt>>>,
    ) -> Self {
        SendRecParam {
            config,
            proc,
            idle,
            waiting,
            _obj: PhantomData,
            _opt: PhantomData,
        }
    }
}

/// A structure containing utilitaries to send [`Partial`] to workers while using multi-threading.
pub struct ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    pub config: Configuration,
    pub proc: &'a MPIProcess,
    pub idle: Arc<Mutex<Vec<usize>>>,
    pub waiting: ArcMutexHash<PSol, PSol::Twin<Opt>, SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
}

impl<'a, PSol, Obj, Opt, SolId, SInfo> ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    pub fn new(
        config: Configuration,
        proc: &'a MPIProcess,
        idle: Arc<Mutex<Vec<usize>>>,
        waiting: ArcMutexHash<PSol, PSol::Twin<Opt>, SolId>,
    ) -> Self {
        ThrSendRecParam {
            config,
            proc,
            idle,
            waiting,
            _obj: PhantomData,
            _opt: PhantomData,
        }
    }
}

/// Send an Obj [`Solution`] to a worker
pub fn send_to_worker<'a, PSol, Obj, Opt, SolId, SInfo>(
    params: &mut SendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
    pair: PartPair<PSol, PSol::Twin<Opt>>,
) -> bool
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    let has_idl = !params.idle.is_empty();
    if has_idl {
        let sid = pair.0.get_id();
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
pub fn fill_workers<'a, PSol, Obj, Opt, SolId, SInfo, Info, St>(
    params: &mut SendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
    stop: Arc<Mutex<St>>,
    batch: &Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    idx: usize,
) -> usize
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    St: Stop,
{
    let mut i: usize = idx;
    let mut st = stop.lock().unwrap();
    let mut at_least_one_idle = true;
    while at_least_one_idle && i < batch.size() && !st.stop() {
        let has_idle = send_to_worker(params, batch.index(i));
        if has_idle {
            st.update(ExpStep::Distribution);
            i += 1;
        } else {
            at_least_one_idle = false
        }
    }
    i
}

pub fn par_send_to_worker<'a, PSol, Obj, Opt, SolId, SInfo>(
    params: &ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
    pair: PartPair<PSol, PSol::Twin<Opt>>,
) -> bool
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
{
    let mut idl = params.idle.lock().unwrap();
    let has_idl = !idl.is_empty();
    if has_idl {
        let sid = pair.0.get_id();
        let rank = idl.pop().unwrap().as_();
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
pub fn par_fill_workers<'a, PSol, Obj, Opt, SolId, SInfo, Info, St>(
    params: &ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
    stop: Arc<Mutex<St>>,
    batch: &Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
    idx: Arc<Mutex<Vec<usize>>>,
) where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    St: Stop,
{
    let mut st = stop.lock().unwrap();
    let mut at_least_one_idle = true;
    let mut idx_list = idx.lock().unwrap();
    while at_least_one_idle && !idx_list.is_empty() && !st.stop() {
        let i = idx_list.pop().unwrap();
        let has_idle = par_send_to_worker(params, batch.index(i));
        if has_idle {
            st.update(ExpStep::Distribution);
        } else {
            at_least_one_idle = false
        }
    }
}

pub fn receive_obj_computed<'a, PSol, Obj, Opt, SolId, SInfo, Info, Cod, Out>(
    params: &mut SendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
    res: &mut BatchResults<PSol, SolId, Obj, Opt, Cod, Out, SInfo, Info>,
    ob: &Objective<Obj, Cod, Out>,
) where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
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
