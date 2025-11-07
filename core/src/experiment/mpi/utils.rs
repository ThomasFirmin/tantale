use crate::{
    Codomain, Computed, Domain, Fidelity, Id, OptInfo, Outcome, Partial, SolInfo,
    experiment::{mpi::tools::MPIProcess, utils::{BatchResults, PartPair}},
    solution::{Batch, partial::FidelityPartial}, stop::{ExpStep, Stop}
};

use bincode::{config::Configuration, serde::Compat};
use mpi::traits::{Communicator, Destination, Source};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

  //-----------------//
 // --- MESSAGES ---//
//-----------------//

/// Describe a message sent to a [`Worker`] and containing a raw solution
/// to be evaluated by a [`Worker`] .
pub trait XMsg<PSol, SolId, Dom, SInfo>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
    SolId:Id,
    Dom:Domain,
    PSol: Partial<SolId, Dom, SInfo>,
    SInfo:SolInfo,
{
    fn new(sol: &PSol) -> Self;
    fn to_bytes(self, config:Configuration)->Vec<u8>{
        bincode::encode_to_vec(Compat(self), config).unwrap()
    }
}

/// A [`XMessage`] is a [`XMsg`] describing the content sent to a [`Worker`].
/// It is made of the raw solution, and its [`Id`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, SolId:Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct XMessage<SolId: Id, Dom: Domain>(pub SolId, pub Arc<[Dom::TypeDom]>);
impl<PSol, SolId, Dom, SInfo> XMsg<PSol, SolId, Dom, SInfo> for XMessage<SolId,Dom>
where
    SolId:Id,
    Dom:Domain,
    PSol: Partial<SolId, Dom, SInfo>,
    SInfo:SolInfo,
{
    fn new(sol: &PSol) -> Self {
        XMessage(sol.get_id(), sol.get_x())
    }
}

/// A [`FXMessage`] is a [`XMsg`] describing the content sent to a [`Worker`].
/// It is made of the raw solution, its [`Id`], and a  [`Fidelity`].
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, SolId:Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct FXMessage<SolId: Id, Dom: Domain>(pub SolId, pub Arc<[Dom::TypeDom]>, pub Fidelity);
impl<PSol, SolId, Dom, SInfo> XMsg<PSol, SolId, Dom, SInfo> for FXMessage<SolId,Dom>
where
    SolId:Id,
    Dom:Domain,
    PSol: FidelityPartial<SolId, Dom, SInfo>,
    SInfo:SolInfo,
{
    fn new(sol: &PSol) -> Self {
        FXMessage(sol.get_id(), sol.get_x(), sol.get_fidelity())
    }
}

/// A structure containing utilitaries to send and [`Partial`] to [`Worker`], and receive [`RawSol`]
/// and [`Computed`] from [`Worker`].
pub struct SendRec<'a, Msg, PSol, Obj, Opt, SolId, SInfo>
where
    Msg: XMsg<PSol,SolId,Obj,SInfo>,
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
    _msg: PhantomData<Msg>,
}

impl<'a, Msg, PSol, Obj, Opt, SolId, SInfo> SendRec<'a, Msg, PSol, Obj, Opt, SolId, SInfo>
where
    Msg: XMsg<PSol,SolId,Obj,SInfo>,
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
        SendRec {config,proc,idle,waiting, _msg: PhantomData}
    }

    /// Send an Obj [`Solution`] to a worker
    pub fn send_to_worker(&mut self, pair: PartPair<PSol, PSol::Twin<Opt>>) -> bool
    where
        PSol: Partial<SolId, Obj, SInfo>,
        PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
        Opt: Domain,
        SolId: Id,
        SInfo: SolInfo,
    {
        let has_idl = !self.idle.is_empty();
        if has_idl {
            let sid = pair.0.get_id();
            let rank = self.idle.pop().unwrap();
            let msg = Msg::new(pair.0.as_ref()).to_bytes(self.config);
            self.waiting.insert(sid, pair);
            self.proc.world.process_at_rank(rank).send_with_tag(&msg, 1);
        }
        has_idl
    }

    // Send as much solutions as possible to idle workers without waiting for a result.
    pub fn fill_workers<Info:OptInfo, St:Stop>(
        &mut self,
        stop: &mut St,
        batch: &Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
        idx: usize,
    ) -> usize
    {
        let mut i: usize = idx;
        let mut at_least_one_idle = true;
        while at_least_one_idle && i < batch.size() && !stop.stop() {
            let has_idle = self.send_to_worker(batch.index(i));
            if has_idle {
                stop.update(ExpStep::Distribution);
                i += 1;
            } else {
                at_least_one_idle = false
            }
        }
        i
    }

    pub fn rec_computed<Info:OptInfo, Cod: Codomain<Out>, Out: Outcome>(
        &mut self,
        res: &mut BatchResults<PSol, SolId, Obj, Opt, Cod, Out, SInfo, Info>,
        cod: &Cod,
    )
    {
        // Recv / sendv loop
        let (bytes, status): (Vec<u8>, _) = self.proc.world.any_process().receive_vec();
        self.idle.push(status.source_rank());
        let (bytes, _): (Compat<OMessage<SolId, Out>>, _) =
            bincode::decode_from_slice(bytes.as_slice(), self.config).unwrap();
        let msg = bytes.0;
        let id = msg.0;
        let out = msg.1;
        let y = Arc::new(cod.get_elem(&out));
        let out = Arc::new(out);
        let pair = self.waiting.remove(&id).unwrap();
        res.add(pair, out, y);
    }
}

pub type VecArcComputed<PSol, SolId, Dom, Cod, Out, SInfo> =
    Vec<Arc<Computed<PSol, SolId, Dom, Cod, Out, SInfo>>>;

pub type ArcMutexHash<PSolA, PSolB, SolId> = Arc<Mutex<HashMap<SolId, PartPair<PSolA, PSolB>>>>;

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId:Serialize",
    deserialize = "SolId:for<'a> Deserialize<'a>",
))]
pub struct DiscardFXMessage<SolId: Id>(pub SolId);


#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct OMessage<SolId: Id, Out: Outcome>(pub SolId, pub Out);

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Out: Serialize, SolId:Serialize",
    deserialize = "Out: for<'a> Deserialize<'a>, SolId:for<'a> Deserialize<'a>",
))]
pub struct FOMessage<SolId: Id, Out: Outcome>(pub SolId, pub Out, pub Fidelity);

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

// pub fn par_send_to_worker<'a, PSol, Obj, Opt, SolId, SInfo>(
//     params: &ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
//     pair: PartPair<PSol, PSol::Twin<Opt>>,
// ) -> bool
// where
//     PSol: Partial<SolId, Obj, SInfo>,
//     PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
//     Obj: Domain,
//     Opt: Domain,
//     SolId: Id,
//     SInfo: SolInfo,
// {
//     let mut idl = params.idle.lock().unwrap();
//     let has_idl = !idl.is_empty();
//     if has_idl {
//         let sid = pair.0.get_id();
//         let rank = idl.pop().unwrap().as_();
//         let x = pair.0.get_x();
//         let msg_struct = Compat((sid, x.as_ref()));
//         let msg = bincode::encode_to_vec(msg_struct, params.config).unwrap();
//         params.waiting.lock().unwrap().insert(sid, pair);
//         params
//             .proc
//             .world
//             .process_at_rank(rank)
//             .send_with_tag(&msg, 1);
//     }
//     has_idl
// }

// // Send as much solutions as possible to idle workers without waiting for a result.
// pub fn par_fill_workers<'a, PSol, Obj, Opt, SolId, SInfo, Info, St>(
//     params: &ThrSendRecParam<'a, PSol, Obj, Opt, SolId, SInfo>,
//     stop: Arc<Mutex<St>>,
//     batch: &Batch<PSol, SolId, Obj, Opt, SInfo, Info>,
//     idx: Arc<Mutex<Vec<usize>>>,
// ) where
//     PSol: Partial<SolId, Obj, SInfo>,
//     PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
//     Obj: Domain,
//     Opt: Domain,
//     SolId: Id,
//     SInfo: SolInfo,
//     Info: OptInfo,
//     St: Stop,
// {
//     let mut st = stop.lock().unwrap();
//     let mut at_least_one_idle = true;
//     let mut idx_list = idx.lock().unwrap();
//     while at_least_one_idle && !idx_list.is_empty() && !st.stop() {
//         let i = idx_list.pop().unwrap();
//         let has_idle = par_send_to_worker(params, batch.index(i));
//         if has_idle {
//             st.update(ExpStep::Distribution);
//         } else {
//             at_least_one_idle = false
//         }
//     }
// }
